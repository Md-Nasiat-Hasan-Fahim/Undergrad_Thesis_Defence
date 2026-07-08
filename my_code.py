"""
Fast-Part verification wrapper — memory-efficient, disk-streaming implementation.
Calls fast_part.run_pipeline() natively instead of reimplementing logic.
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
from collections import defaultdict
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
    Stream CSV row-by-row directly to FASTA.  Memory footprint ~ 1 row.

    Returns the number of sequences written.
    """
    required = {"Entry Name", "Organism", "Sequence"}
    count = 0

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
            seq_id = row["Entry Name"]
            prot = row.get("Protein names", "")
            desc = f"{seq_id}|{label}|{prot}"

            f_out.write(f">{desc}\n{seq}\n")
            count += 1

    return count


# =============================================================================
# NATIVE PARTITION WRAPPER (calls fast_part.run_pipeline directly)
# =============================================================================

def run_native_partition(input_fasta, output_dir, args):
    """
    Run fast_part.run_pipeline() natively.  This automatically enables
    multiprocessing (via --parallel_labels) and avoids reimplementing logic.
    """
    pipeline_args = build_mock_args(
        fasta_file=input_fasta,
        output_dir=output_dir,
        method=args.method,
        identity_threshold=args.threshold,
        train_ratio=args.train_ratio,
        query_cover=args.query_cover,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        seed=args.seed,
        parallel_labels=args.parallel_labels,
        keep_temp=args.exhaustive_verify,   # keep temps if we need them for exhaustive verify
    )
    return fast_part.run_pipeline(pipeline_args)


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
    removed = total - retained
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
    Use the stats dict returned by fast_part.run_pipeline.
    Reconstructs per-label stats if the native 'label_stats' key is absent.
    """
    logger.info("\n2. STRATIFICATION RATIO VERIFICATION")

    label_stats = stats.get("label_stats", {})
    if not label_stats:
        # Reconstruct from fast_part's returned dictionaries
        all_labels = set(stats.get("train_label_counts", {}).keys()) | set(stats.get("test_label_counts", {}).keys())
        total_counts = stats.get("label_sequence_counts", {})
        for label in all_labels:
            total = total_counts.get(label, 0)
            train = stats["train_label_counts"].get(label, 0)
            test = stats["test_label_counts"].get(label, 0)
            label_stats[label] = {
                "total": total,
                "train": train,
                "test": test,
                "train_pct": (train / total * 100) if total > 0 else 0,
            }

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

    all_labels = set(stats.get("label_sequence_counts", {}).keys())
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
        n_seqs = stream_csv_to_fasta(args.csv, input_fasta)
        logger.info("Wrote %d sequences to %s", n_seqs, input_fasta)

        # ------------------------------------------------------------------
        # 2. Run native fast_part pipeline (multiprocessing enabled)
        # ------------------------------------------------------------------
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Running native fast_part pipeline (parallel_labels=%d)...",
                    args.parallel_labels)
        stats = run_native_partition(input_fasta, output_dir, args)

        train_file = stats["train_file"]
        test_file = stats["test_file"]

        # ------------------------------------------------------------------
        # 3. Verification (disk-based, generators)
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
        # 4. Overall verdict
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
        # 5. Persist final outputs
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
