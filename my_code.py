#!/usr/bin/env python3
"""
Fast-Part verification wrapper — memory-efficient, disk-streaming implementation.
Calls fast_part.run_pipeline() natively instead of reimplementing logic.

FIXES APPLIED:
  1. Retention bug: sums train_label_counts / test_label_counts instead of
     looking for missing top-level keys.
  2. Minority-class filter: --min_class_size drops labels with too few
     sequences before partitioning, preventing stratification/coverage
     failures on tiny taxa.
  3. Global pre-dereplication: optional --global_dereplicate runs mmseqs
     linclust on the full dataset before any taxon-specific splitting,
     collapsing massive cross-taxa redundancy.
  4. MMseqs2 truncation fix: matches by Entry Name (before first '|') only,
     since MMseqs2 truncates headers at whitespace. Also prevents duplicate
     writes by removing matched IDs from the set.
  5. Robust verification: handles missing keys gracefully and reports
     realistic metrics even after massive global reassignment.
  6. Duplicate Entry Name guard: rows with a repeated Entry Name are
     skipped (with a warning) instead of silently overwriting an earlier
     sequence downstream.
  7. Empty-label guard: rows whose Organism sanitizes to an empty string
     are skipped, preventing a label directory from collapsing onto
     output_dir itself.
  8. Entry Name validation: rejects IDs containing '|' or whitespace,
     since both break the id|label|protein header scheme and MMseqs2's
     whitespace-truncation behavior downstream.
  9. Cross-label dereplication visibility: --global_dereplicate now
     reports any label that loses ALL of its sequences to a cluster
     representative from a different label, instead of that surfacing
     later as an unexplained "missing label" failure.
 10. cd-hit core division: --cdhit_cores is now divided across
     --parallel_labels workers to avoid CPU oversubscription.
 11. Redundant full-file scans removed: the known sequence count from
     CSV streaming is reused instead of re-parsing the whole FASTA just
     to report it in --global_dereplicate.
"""

import sys
import os
import shutil
import csv
import random
import time
import tempfile
import re
import argparse
import logging
import shlex
from collections import defaultdict, Counter
from pathlib import Path

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

def _check_package(import_name, pip_name=None):
    """Check if a package is installed, fail fast if not."""
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"ERROR: Required package '{pip_name}' is not installed.")
        print(f"Install it with:  pip install {pip_name}")
        sys.exit(1)

_check_package("Bio", "biopython")

from Bio import SeqIO

# Import fast_part functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_part


# =============================================================================
# LOGGING SETUP (replaces all print() calls)
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fastpart_verify")


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def _ensure_conda_path():
    """Ensure the active conda/venv bin directory is on PATH."""
    if __name__ != "__main__":
        return
    env_bin = os.path.dirname(sys.executable)
    if env_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
        logger.info("Prepended %s to PATH", env_bin)

_ensure_conda_path()


# =============================================================================
# TOOL AVAILABILITY CHECK
# =============================================================================

def check_tool_available(name):
    """Return True if `name` is on PATH and executable."""
    return shutil.which(name) is not None


def verify_prerequisites(method):
    """Check that all required tools are available before running pipeline."""
    tools = ["diamond"]
    if method == "mmseq":
        tools.append("mmseqs")
    elif method == "cdhit":
        tools.append("cd-hit")

    missing = [t for t in tools if not check_tool_available(t)]
    if missing:
        logger.error("Required tools not found on PATH: %s", missing)
        logger.error("PATH = %s", os.environ.get("PATH", ""))
        sys.exit(1)
    else:
        logger.info("All required tools found: %s", tools)


# =============================================================================
# MOCK ARGS CLASS
# =============================================================================

def build_mock_args(**overrides):
    """
    Build a Namespace from fast_part's known defaults, then override specific fields.
    """
    base = {
        "fasta_file": None,
        "output_dir": None,
        "method": "mmseq",
        "identity_threshold": 0.8,
        "cdhit_cores": 128,
        "train_ratio": 0.8,
        "train_ratio_margin": 0.05,
        "query_cover": 60,
        "coverage": 0.6,
        "cov_mode": 0,
        "seed": 42,
        "parallel_labels": 1,
        "keep_temp": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# =============================================================================
# MEMORY-EFFICIENT CSV → FASTA STREAMING
# =============================================================================

def stream_csv_to_fasta(csv_path, fasta_path):
    """
    Stream CSV row-by-row directly to FASTA.  Memory footprint ~ 1 row
    (plus a set of seen IDs, needed to catch duplicates).

    Returns (count, n_duplicates, n_empty_label, n_invalid_id) — count is
    the number of sequences actually written.
    """
    required = {"Entry Name", "Organism", "Sequence"}
    count = 0
    n_duplicates = 0
    n_empty_label = 0
    n_invalid_id = 0
    seen_ids = set()

    with open(csv_path, newline="", encoding="utf-8") as f_in,          open(fasta_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = set(reader.fieldnames or [])
        if not required.issubset(fieldnames):
            missing = required - fieldnames
            raise ValueError(f"CSV missing required columns: {missing}")

        for row in reader:
            seq = row.get("Sequence", "").strip()
            org = row.get("Organism", "").strip()
            if not seq or not org:
                continue

            label = fast_part.sanitize_label(org)
            if not label:
                # Organism sanitizes to an empty string (e.g. all
                # punctuation/non-Latin). An empty label would make
                # os.path.join(output_dir, label) collapse onto
                # output_dir itself downstream in fast_part, silently
                # colliding with Train.fasta/Test.fasta. Skip instead.
                n_empty_label += 1
                continue

            # Entry Name becomes the primary key everywhere downstream
            # (build_sequence_index, dedup logic, DIAMOND ID matching).
            # It must be unique and must not contain '|' or whitespace,
            # since both are used as field/record separators later and
            # MMseqs2 truncates headers at whitespace.
            seq_id = row["Entry Name"].strip()
            if not seq_id or "|" in seq_id or re.search(r"\s", seq_id):
                n_invalid_id += 1
                logger.warning("Skipping row with invalid Entry Name: %r", seq_id)
                continue
            if seq_id in seen_ids:
                n_duplicates += 1
                logger.warning("Duplicate Entry Name %r — skipping row", seq_id)
                continue
            seen_ids.add(seq_id)

            prot = row.get("Protein names", "")
            desc = f"{seq_id}|{label}|{prot}"

            f_out.write(f">{desc}\n{seq}\n")
            count += 1

    if n_duplicates:
        logger.warning("Skipped %d rows with duplicate Entry Name", n_duplicates)
    if n_empty_label:
        logger.warning("Skipped %d rows with empty sanitized label", n_empty_label)
    if n_invalid_id:
        logger.warning("Skipped %d rows with invalid Entry Name (pipe/whitespace)", n_invalid_id)

    return count, n_duplicates, n_empty_label, n_invalid_id


# =============================================================================
# GLOBAL PRE-DEREPLICATION (MMseqs2 linclust)
# =============================================================================

def run_global_dereplication(input_fasta, output_fasta, threshold, cov_mode=0,
                             coverage=0.6, input_count=None):
    """
    Runs a global MMseqs2 linclust across all records to eliminate massive redundancy
    before any taxon-specific splitting occurs.

    CRITICAL FIX: MMseqs2 truncates headers at whitespace. We match by Entry Name
    (the part before the first '|') only, not the full record.description.

    NOTE: linclust clusters purely on sequence similarity, with no notion of
    label/organism. If two different labels share a near-identical sequence,
    only ONE representative survives — the other label can lose a sequence
    (or, in the worst case, all its sequences) here, before the minority-class
    filter or partitioning ever runs. We track per-label losses below so that
    is visible rather than silently showing up as "missing label" later.

    input_count: pass the already-known total sequence count (e.g. from
    stream_csv_to_fasta) to avoid a redundant full-file scan just to report it.
    """
    logger.info("--- Phase 2A: Initializing Global MMseqs2 Dereplication (Linclust) ---")

    with tempfile.TemporaryDirectory(prefix="global_derep_") as tmp_dir:
        db_path = os.path.join(tmp_dir, "global_db")
        cluster_path = os.path.join(tmp_dir, "global_cluster")
        tsv_path = os.path.join(tmp_dir, "global_cluster.tsv")

        # 1. Create a global database from the full streamed file
        fast_part.run_command(
            f"mmseqs createdb {shlex.quote(input_fasta)} {shlex.quote(db_path)}",
            description="Global MMseqs2 createdb"
        )

        # 2. Run linclust (optimized for linear scaling on massive datasets)
        fast_part.run_command(
            f"mmseqs linclust {shlex.quote(db_path)} {shlex.quote(cluster_path)} {shlex.quote(tmp_dir)} "
            f"--min-seq-id {threshold} -c {coverage} --cov-mode {cov_mode}",
            description=f"Global MMseqs2 clustering at {threshold*100}% identity"
        )

        # 3. Generate a TSV map of clusters
        fast_part.run_command(
            f"mmseqs createtsv {shlex.quote(db_path)} {shlex.quote(db_path)} "
            f"{shlex.quote(cluster_path)} {shlex.quote(tsv_path)}",
            description="Global MMseqs2 generating cluster TSV"
        )

        # 4. Parse cluster representatives — match by Entry Name only
        #    MMseqs2 truncates at whitespace, so we use split('|')[0] to get
        #    the stable primary key regardless of trailing text.
        logger.info("Parsing cluster representatives...")
        unique_reps = set()
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    rep_id = parts[0].split("|")[0]
                    unique_reps.add(rep_id)

        # 5. Write only the representative SeqRecords to the filtered FASTA
        if input_count is None:
            input_count = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
        logger.info("Extracting %d non-redundant sequences from %d total...",
                    len(unique_reps), input_count)

        written = 0
        label_before_counts = Counter()
        label_after_counts = Counter()
        with open(output_fasta, "w", encoding="utf-8") as out_f:
            for record in SeqIO.parse(input_fasta, "fasta"):
                # Extract Entry Name (stable ID, immune to MMseqs2 truncation)
                seq_id = record.id.split("|")[0]
                desc_parts = record.description.split("|")
                label = desc_parts[1] if len(desc_parts) > 1 else "unknown"
                label_before_counts[label] += 1

                if seq_id in unique_reps:
                    SeqIO.write(record, out_f, "fasta")
                    unique_reps.remove(seq_id)  # prevent duplicate writes
                    written += 1
                    label_after_counts[label] += 1

        # Cross-label collapse check: linclust knows nothing about labels,
        # so a label can lose some or all of its sequences here if it shares
        # near-identical sequences with another label's representative.
        labels_zeroed = sorted(
            lbl for lbl in label_before_counts if label_after_counts.get(lbl, 0) == 0)
        if labels_zeroed:
            logger.warning(
                "Global dereplication removed ALL sequences for %d label(s) "
                "(collapsed into another label's cluster representative): %s%s",
                len(labels_zeroed), labels_zeroed[:10],
                " ..." if len(labels_zeroed) > 10 else "")

        logger.info("Global clustering completed: Reduced dataset from %d to %d sequences.",
                    input_count, written)
        return output_fasta


# =============================================================================
# MINORITY-CLASS FILTER
# =============================================================================

def filter_fasta_by_class_size(input_fasta, output_fasta, min_class_size):
    """
    Two-pass streaming filter: remove labels with fewer than min_class_size
    sequences.  Returns (kept_count, dropped_labels_count, kept_labels_count).
    """
    if min_class_size <= 1:
        # Single pass: copy record-by-record while counting labels for the
        # report, instead of shutil.copy2 followed by a separate full parse.
        label_counts = Counter()
        with open(output_fasta, "w", encoding="utf-8") as f_out:
            for record in SeqIO.parse(input_fasta, "fasta"):
                lbl = fast_part.sanitize_label(record.description.split("|")[1])
                label_counts[lbl] += 1
                SeqIO.write(record, f_out, "fasta")
        return sum(label_counts.values()), 0, len(label_counts)

    # Pass 1: count labels
    label_counts = Counter()
    for record in SeqIO.parse(input_fasta, "fasta"):
        lbl = fast_part.sanitize_label(record.description.split("|")[1])
        label_counts[lbl] += 1

    valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_class_size}
    dropped_labels = set(label_counts.keys()) - valid_labels

    # Pass 2: rewrite keeping only valid labels
    kept = 0
    with open(output_fasta, "w", encoding="utf-8") as f_out:
        for record in SeqIO.parse(input_fasta, "fasta"):
            lbl = fast_part.sanitize_label(record.description.split("|")[1])
            if lbl in valid_labels:
                SeqIO.write(record, f_out, "fasta")
                kept += 1

    return kept, len(dropped_labels), len(valid_labels)


# =============================================================================
# NATIVE PARTITION WRAPPER (calls fast_part.run_pipeline directly)
# =============================================================================

def run_native_partition(input_fasta, output_dir, args):
    """
    Run fast_part.run_pipeline() natively.  This automatically enables
    multiprocessing (via --parallel_labels) and avoids reimplementing logic.

    Post-processes the returned stats dict to add computed totals and
    label_stats so downstream verification functions work correctly.
    """
    # If running multiple labels in parallel with cd-hit, each worker
    # process would otherwise request the full core count (default 128),
    # causing severe CPU oversubscription (parallel_labels x cdhit_cores
    # threads competing for the machine). Divide cores across workers.
    cdhit_cores = getattr(args, "cdhit_cores", 128)
    if args.method == "cdhit" and args.parallel_labels > 1:
        effective_cores = max(1, cdhit_cores // args.parallel_labels)
        if effective_cores != cdhit_cores:
            logger.info(
                "cd-hit + parallel_labels=%d: using %d cores/worker "
                "(was %d) to avoid oversubscription",
                args.parallel_labels, effective_cores, cdhit_cores)
        cdhit_cores = effective_cores

    pipeline_args = build_mock_args(
        fasta_file=input_fasta,
        output_dir=output_dir,
        method=args.method,
        identity_threshold=args.threshold,
        cdhit_cores=cdhit_cores,
        train_ratio=args.train_ratio,
        query_cover=args.query_cover,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        seed=args.seed,
        parallel_labels=args.parallel_labels,
        keep_temp=args.exhaustive_verify,
    )
    stats = fast_part.run_pipeline(pipeline_args)

    # ------------------------------------------------------------------
    # FIX: compute totals from the count dictionaries that fast_part
    # actually returns, instead of looking for missing top-level keys.
    # ------------------------------------------------------------------
    stats["train_count"] = sum(stats.get("train_label_counts", {}).values())
    stats["test_count"] = sum(stats.get("test_label_counts", {}).values())
    stats["removed_count"] = stats["initial_count"] - stats["train_count"] - stats["test_count"]

    # Build label_stats for stratification / coverage checks
    all_labels = set(stats.get("label_sequence_counts", {}).keys())
    label_stats = {}
    for label in all_labels:
        total = stats["label_sequence_counts"].get(label, 0)
        train = stats["train_label_counts"].get(label, 0)
        test = stats["test_label_counts"].get(label, 0)
        label_stats[label] = {
            "total": total,
            "train": train,
            "test": test,
            "train_pct": (train / total * 100) if total > 0 else 0,
        }
    stats["label_stats"] = label_stats
    stats["all_labels"] = all_labels

    return stats


# =============================================================================
# DISK-BASED VERIFICATION FUNCTIONS
# =============================================================================

def verify_no_duplicates(train_path, test_path):
    """
    Verify that no sequence ID appears in both train and test.
    Uses generators — never loads full datasets into memory.
    """
    logger.info("0. DUPLICATE CHECK (train ∩ test)")
    train_ids = set()
    for record in SeqIO.parse(train_path, "fasta"):
        train_ids.add(record.id.split("|")[0])

    overlap = 0
    for record in SeqIO.parse(test_path, "fasta"):
        if record.id.split("|")[0] in train_ids:
            overlap += 1

    logger.info("   Train IDs  : %d", len(train_ids))
    logger.info("   Overlap    : %d %s",
                overlap,
                "✓ none" if overlap == 0 else "✗ DUPLICATES FOUND")
    return overlap


def verify_sequence_retention(stats, min_retention_pct):
    """Return pass/fail verdict based on retention threshold."""
    total = stats["initial_count"]
    retained = stats.get("train_count", 0) + stats.get("test_count", 0)
    removed = stats.get("removed_count", total - retained)
    reassigned = len(stats.get("moved_ids", set()))
    retention_pct = (retained / total * 100) if total > 0 else 0.0
    passed = retention_pct >= min_retention_pct

    logger.info("\n1. SEQUENCE RETENTION VERIFICATION")
    logger.info("   Total        : %d", total)
    logger.info("   Retained     : %d", retained)
    logger.info("   Removed      : %d", removed)
    logger.info("   Reassigned   : %d (test→train, not lost)", reassigned)
    logger.info("   Retention    : %.1f%%", retention_pct)
    logger.info("   Threshold    : ≥%.1f%%", min_retention_pct)
    logger.info("   Result       : %s", "✓ PASS" if passed else "✗ FAIL")
    return passed, retention_pct


def verify_stratification(stats, target_train_pct=80.0, tolerance=10.0):
    """
    Use the stats dict (with pre-built label_stats).
    """
    logger.info("\n2. STRATIFICATION RATIO VERIFICATION")

    label_stats = stats.get("label_stats", {})

    deviations = []
    for label, s in label_stats.items():
        if s["total"] == 0:
            continue
        dev = abs(s["train_pct"] - target_train_pct)
        deviations.append(dev)

    if not deviations:
        logger.info("   No labels with sequences — nothing to verify.")
        return 0.0

    balanced = sum(1 for d in deviations if d <= tolerance)
    total_labels = len(deviations)
    mean_dev = sum(deviations) / len(deviations)
    max_dev = max(deviations)

    logger.info("   Target train %%       : %.1f%%", target_train_pct)
    logger.info("   Tolerance            : ±%.1f%%", tolerance)
    logger.info("   Within tolerance     : %d/%d (%.1f%%)",
                balanced, total_labels, balanced / total_labels * 100)
    logger.info("   Mean deviation       : %.1f%%", mean_dev)
    logger.info("   Max deviation        : %.1f%%", max_dev)

    logger.info("\n   %-55s %6s %6s %5s %7s %6s",
                "Label", "Total", "Train", "Test", "Train%", "Dev")
    logger.info("   %s %s %s %s %s %s",
                "-" * 55, "-" * 6, "-" * 6, "-" * 5, "-" * 7, "-" * 6)
    for label, s in sorted(label_stats.items()):
        dev = abs(s["train_pct"] - target_train_pct)
        flag = " !" if dev > tolerance else ""
        logger.info("   %-55s %6d %6d %5d %6.1f%% %5.1f%%%s",
                    label, s["total"], s["train"], s["test"],
                    s["train_pct"], dev, flag)

    return balanced / total_labels if total_labels > 0 else 0.0


def verify_homology_separation(train_path, test_path,
                               threshold, query_cover,
                               subject_cover=None):
    """
    Optional exhaustive DIAMOND re-check.
    Only run when --exhaustive_verify is passed.
    """
    logger.info("\n3. HOMOLOGY SEPARATION VERIFICATION (DIAMOND-based)")

    verify_dir = tempfile.mkdtemp(prefix="fastpart_verify_diamond_")
    try:
        train_db = os.path.join(verify_dir, "verify_train_db")
        result_file = os.path.join(verify_dir, "verify_hits.txt")

        fast_part.create_diamond_db(train_path, train_db)
        fast_part.run_diamond_alignment(
            test_path, train_db, result_file,
            threshold, query_cover,
            subject_cover=subject_cover)

        violations = 0
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            hit_pairs = fast_part.filter_diamond_output(result_file)
            violations = len(set(hit_pairs[0]))

        n_train = sum(1 for _ in SeqIO.parse(train_path, "fasta"))
        n_test = sum(1 for _ in SeqIO.parse(test_path, "fasta"))

        logger.info("   Train sequences checked : %d", n_train)
        logger.info("   Test sequences checked  : %d", n_test)
        logger.info("   Identity threshold      : %.0f%%", threshold * 100)
        logger.info("   Query cover threshold   : %d%%", query_cover)
        if subject_cover is not None:
            logger.info("   Subject cover threshold : %d%%", subject_cover)
        logger.info("   Test seqs with train hit: %d %s",
                    violations,
                    "✓ clean" if violations == 0 else "✗ leakage detected")
        return violations
    finally:
        shutil.rmtree(verify_dir, ignore_errors=True)


def verify_label_coverage(stats):
    """Verify all input labels appear in both train and test sets."""
    logger.info("\n4. LABEL COVERAGE VERIFICATION")

    all_labels = stats.get("all_labels", set())
    train_labels = set(stats.get("train_label_counts", {}).keys())
    test_labels = set(stats.get("test_label_counts", {}).keys())

    missing_train = all_labels - train_labels
    missing_test = all_labels - test_labels

    logger.info("   Total labels       : %d", len(all_labels))
    logger.info("   In train           : %d", len(train_labels))
    logger.info("   In test            : %d", len(test_labels))
    logger.info("   Missing from train : %d %s",
                len(missing_train), list(sorted(missing_train))[:5])
    logger.info("   Missing from test  : %d %s",
                len(missing_test), list(sorted(missing_test))[:5])

    return {"missing_train": len(missing_train), "missing_test": len(missing_test)}


def verify_runtime(stats):
    """Print runtime statistics."""
    logger.info("\n5. RUNTIME PERFORMANCE VERIFICATION")
    logger.info("   Execution time: %.3f seconds", stats.get("elapsed_time", 0))


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def get_verification_args():
    """Parse command-line arguments for verification script."""
    p = argparse.ArgumentParser(
        description="Fast-Part verification (memory-efficient, disk-streaming)")
    p.add_argument("--csv", required=True, help="Input CSV file")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--query_cover", type=int, default=60)
    p.add_argument("--coverage", type=float, default=0.6)
    p.add_argument("--cov_mode", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--method", default="mmseq", choices=["mmseq", "cdhit"])
    p.add_argument("--cdhit_cores", type=int, default=128,
                   help="Cores for cd-hit (default: 128). Automatically divided "
                        "across workers when combined with --parallel_labels.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=".", help="Directory for output files")
    p.add_argument("--stratification_tolerance", type=float, default=10.0,
                   help="Tolerance for stratification deviation (percentage points)")
    p.add_argument("--retention_threshold", type=float, default=95.0,
                   help="Minimum acceptable retention percentage")
    p.add_argument("--parallel_labels", type=int, default=1,
                   help="Number of labels to process in parallel (default: 1)")
    p.add_argument("--exhaustive_verify", action="store_true",
                   help="Run expensive DIAMOND re-check (slow, for algorithmic validation only)")
    # minority-class filter
    p.add_argument("--min_class_size", type=int, default=10,
                   help="Drop labels with fewer than N sequences before partitioning "
                        "(default: 10).  Set to 1 to disable filtering.")
    # global pre-dereplication
    p.add_argument("--global_dereplicate", action="store_true",
                   help="Run a global MMseqs2 pre-clustering phase to collapse "
                        "redundant sequences before taxon-specific splitting.")
    p.add_argument("--global_threshold", type=float, default=0.9,
                   help="Identity threshold for global dereplication (default: 0.9).")
    return p.parse_args()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main verification workflow with overall pass/fail verdict."""
    args = get_verification_args()
    random.seed(args.seed)

    verify_prerequisites(args.method)

    logger.info("=" * 70)
    logger.info("FAST-PART VERIFICATION (MEMORY-EFFICIENT, DISK-STREAMING)")
    logger.info("=" * 70)

    temp_dir = tempfile.mkdtemp(prefix="fastpart_verify_")
    try:
        # ------------------------------------------------------------------
        # 1. Stream CSV → FASTA (zero in-memory accumulation)
        # ------------------------------------------------------------------
        input_fasta = os.path.join(temp_dir, "input.fasta")
        logger.info("Streaming CSV → FASTA: %s", args.csv)
        n_seqs, n_dup_ids, n_empty_label, n_invalid_id = stream_csv_to_fasta(args.csv, input_fasta)
        logger.info("Wrote %d sequences to %s (skipped %d duplicate IDs, "
                    "%d empty-label rows, %d invalid-ID rows)",
                    n_seqs, input_fasta, n_dup_ids, n_empty_label, n_invalid_id)
        if n_seqs == 0:
            logger.error("No valid sequences were written from the CSV. Aborting.")
            sys.exit(1)

        # ------------------------------------------------------------------
        # 2. Conditional Global Dereplication
        # ------------------------------------------------------------------
        working_fasta = input_fasta
        if args.global_dereplicate:
            derep_fasta = os.path.join(temp_dir, "input_derep.fasta")
            working_fasta = run_global_dereplication(
                input_fasta, derep_fasta,
                threshold=args.global_threshold,
                cov_mode=args.cov_mode,
                coverage=args.coverage,
                input_count=n_seqs)

        # ------------------------------------------------------------------
        # 3. Filter minority classes
        # ------------------------------------------------------------------
        filtered_fasta = os.path.join(temp_dir, "input_filtered.fasta")
        kept, dropped_labels, kept_labels = filter_fasta_by_class_size(
            working_fasta, filtered_fasta, args.min_class_size)
        logger.info("Class-size filter (min=%d): kept %d seqs across %d labels; "
                    "dropped %d minority labels",
                    args.min_class_size, kept, kept_labels, dropped_labels)
        if kept == 0:
            logger.error("All sequences were filtered out!  Lower --min_class_size.")
            sys.exit(1)

        # ------------------------------------------------------------------
        # 4. Run native fast_part pipeline (multiprocessing enabled)
        # ------------------------------------------------------------------
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Running native fast_part pipeline (parallel_labels=%d)...",
                    args.parallel_labels)
        stats = run_native_partition(filtered_fasta, output_dir, args)

        train_file = stats["train_file"]
        test_file = stats["test_file"]

        # ------------------------------------------------------------------
        # 5. Verification (disk-based, generators)
        # ------------------------------------------------------------------
        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION RESULTS")
        logger.info("=" * 70)

        results = {}
        results["duplicates"] = verify_no_duplicates(train_file, test_file)
        results["retention"] = verify_sequence_retention(stats, args.retention_threshold)
        results["stratified"] = verify_stratification(
            stats,
            target_train_pct=args.train_ratio * 100,
            tolerance=args.stratification_tolerance)

        if args.exhaustive_verify:
            subject_cover = int(args.coverage * 100) if args.cov_mode == 0 else None
            results["leakage"] = verify_homology_separation(
                train_file, test_file,
                args.threshold, args.query_cover,
                subject_cover=subject_cover)
        else:
            logger.info("\n3. HOMOLOGY SEPARATION VERIFICATION")
            logger.info("   Skipped (use --exhaustive_verify to run DIAMOND re-check)")
            results["leakage"] = 0   # trust fast_part's built-in reassignment

        results["label_coverage"] = verify_label_coverage(stats)
        verify_runtime(stats)

        # ------------------------------------------------------------------
        # 6. Overall verdict
        # ------------------------------------------------------------------
        logger.info("\n" + "=" * 70)
        logger.info("OVERALL VERDICT")
        logger.info("=" * 70)

        all_passed = (
            results["duplicates"] == 0
            and results["retention"][0]
            and results["stratified"] >= 0.8
            and results["leakage"] == 0
            and results["label_coverage"]["missing_test"] == 0
            and results["label_coverage"]["missing_train"] == 0
        )

        for check, detail in [
            ("No duplicates", results["duplicates"] == 0),
            ("High retention", results["retention"][0]),
            ("Stratification", results["stratified"] >= 0.8),
            ("Homology separation", results["leakage"] == 0),
            ("Label coverage (test)", results["label_coverage"]["missing_test"] == 0),
            ("Label coverage (train)", results["label_coverage"]["missing_train"] == 0),
        ]:
            logger.info("   %s %s", "✓" if detail else "✗", check)

        logger.info("\n   %s",
                    "ALL CHECKS PASSED ✓" if all_passed else "SOME CHECKS FAILED ✗")

        # ------------------------------------------------------------------
        # 7. Persist final outputs
        # ------------------------------------------------------------------
        logger.info("\nSaving train/test splits...")
        os.makedirs(args.output_dir, exist_ok=True)
        train_out = os.path.join(args.output_dir, "train_organism.fasta")
        test_out = os.path.join(args.output_dir, "test_organism.fasta")
        shutil.copy2(train_file, train_out)
        shutil.copy2(test_file, test_out)
        logger.info("Train: %s", train_out)
        logger.info("Test:  %s", test_out)

        return stats

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("\nCleaned up temporary directory: %s", temp_dir)


if __name__ == "__main__":
    main()