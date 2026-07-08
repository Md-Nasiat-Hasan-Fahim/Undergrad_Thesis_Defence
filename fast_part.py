import os
import random
import time
import argparse
import subprocess
import shlex
import shutil
import logging
import re
import heapq
from Bio import SeqIO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fast_part")

def run_command(cmd, description=""):
    """
    Execute a shell command safely with full error handling.
    
    - Raises RuntimeError on non-zero exit code
    - Captures and logs stderr
    - Returns CompletedProcess for inspection
    """
    logger.info(f"Running: {description or cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            logger.debug(f"stderr: {result.stderr.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed (exit {e.returncode}): {cmd}")
        logger.error(f"stderr: {e.stderr.strip()}")
        raise RuntimeError(
            f"External tool failed: {description or cmd}\n"
            f"Exit code: {e.returncode}\n"
            f"stderr: {e.stderr.strip()}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Tool not found. Is it installed and on PATH?\n"
            f"Command: {cmd}"
        ) from e


def validate_file_exists(filepath, tool_name="tool"):
    """Raise a clear error if an expected output file doesn't exist or is empty."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{tool_name} did not produce expected output file: {filepath}\n"
            f"Check that the tool is installed and the input file is valid."
        )
    if os.path.getsize(filepath) == 0:
        logger.warning(f"{tool_name} output file is empty: {filepath}")


def sanitize_label(raw_label):
    """
    Strip ALL non-alphanumeric characters, collapse runs of underscores,
    and strip leading/trailing underscores.
    
    "Pseudomonas putida (Arthrobacter siderocapsulatus)"
    → "Pseudomonas_putida_Arthrobacter_siderocapsulatus"
    """
    label = re.sub(r'[^A-Za-z0-9]', '_', str(raw_label).strip())
    label = re.sub(r'_+', '_', label)
    return label.strip('_')


def build_parser():
    """Return the argument parser without calling parse_args()."""
    parser = argparse.ArgumentParser(description="A tool for efficient partitioning and clustering of sequences.")
    parser.add_argument("--fasta_file", required=True, help="Input FASTA file containing sequences")
    parser.add_argument("--output_dir", required=True, help="Directory to store output files")
    parser.add_argument("--method", required=True, choices=["mmseq", "cdhit"], help="Clustering method: 'mmseq' or 'cdhit'")
    parser.add_argument("--identity_threshold", type=float, default=0.8, help="Identity threshold for clustering")
    parser.add_argument("--cdhit_cores", type=int, default=128, help="Number of cores to use for CD-HIT (if used)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train set ratio for splitting clusters (5% less is used)")
    parser.add_argument("--query_cover", type=int, default=60, help="Query coverage for DIAMOND alignment")
    parser.add_argument("--coverage", type=float, default=0.6,
                        help="Minimum coverage fraction for both clustering and alignment")
    parser.add_argument("--cov_mode", type=int, default=0, choices=[0, 1, 2, 3],
                        help="MMseqs2 coverage mode: 0=bidirectional, 1=target, 2=query")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_ratio_margin", type=float, default=0.05,
                        help="Safety margin subtracted from train_ratio "
                             "to leave room for DIAMOND reassignment (default: 0.05)")
    parser.add_argument("--parallel_labels", type=int, default=1,
                        help="Number of labels to process in parallel (default: 1 = sequential)")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary/intermediate files for debugging")
    return parser

def get_args():
    return build_parser().parse_args()

# Determine word length based on identity threshold for CD-HIT
def get_word_length(identity_threshold):
    if 0.4 <= identity_threshold < 0.5:
        return 2
    elif 0.5 <= identity_threshold < 0.6:
        return 3
    elif 0.6 <= identity_threshold < 0.7:
        return 4
    else:
        return 5

def parse_and_separate_fasta(fasta_file):
    sequences_by_label = defaultdict(list)
    for record in SeqIO.parse(fasta_file, "fasta"):
        raw_label = record.description.split('|')[1]
        label = sanitize_label(raw_label)
        sequences_by_label[label].append(record)
    return sequences_by_label


def write_sequences_by_label(sequences_by_label, output_dir):
    """
    Yield (label, fasta_path) for each label.
    Expects labels to already be sanitized (from parse_and_separate_fasta).
    """
    for label, sequences in list(sequences_by_label.items()):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        output_file = os.path.join(label_dir, f"{label}.fasta")
        SeqIO.write(sequences, output_file, "fasta")
        yield label, output_file

def run_cdhit(input_file, output_file, identity_threshold, cores):
    word_length = get_word_length(identity_threshold)
    cmd = (
        f"cd-hit"
        f" -i {shlex.quote(input_file)}"
        f" -o {shlex.quote(output_file)}"
        f" -c {identity_threshold}"
        f" -n {word_length}"
        f" -T {cores}"
    )
    run_command(cmd, description=f"CD-HIT clustering ({os.path.basename(input_file)})")


def run_mmseqs2_clustering(input_file, output_file, tmp_dir, identity_threshold,
                           coverage=0.6, cov_mode=0):
    """
    Run MMseqs2 clustering.
    
    Parameters
    ----------
    coverage : float
        Minimum coverage fraction (0-1).
    cov_mode : int
        0 = bidirectional, 1 = target, 2 = query, 3 = target-or-query.
        Default changed to 0 (bidirectional) to align with DIAMOND's query-cover
        semantics.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    db_path      = f"{output_file}_db"
    cluster_path = f"{output_file}_cluster"
    tsv_path     = f"{output_file}.tsv"

    run_command(
        f"mmseqs createdb {shlex.quote(input_file)} {shlex.quote(db_path)}",
        description="MMseqs2 createdb",
    )
    run_command(
        f"mmseqs cluster {shlex.quote(db_path)} {shlex.quote(cluster_path)}"
        f" {shlex.quote(tmp_dir)}"
        f" --min-seq-id {identity_threshold}"
        f" -c {coverage}"
        f" --cov-mode {cov_mode}",
        description="MMseqs2 cluster",
    )
    run_command(
        f"mmseqs createtsv {shlex.quote(db_path)} {shlex.quote(db_path)}"
        f" {shlex.quote(cluster_path)} {shlex.quote(tsv_path)}",
        description="MMseqs2 createtsv",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

def parse_clusters(method, cluster_file):
    clusters = defaultdict(list)
    if method == "cdhit":
        with open(cluster_file, 'r') as file:
            current_cluster = None
            for line in file:
                if line.startswith('>Cluster'):
                    current_cluster = line.strip().split()[-1]
                elif current_cluster:
                    if '>' not in line:
                        continue
                    sequence_id = line.split('>')[1].split('|')[0]
                    clusters[current_cluster].append(sequence_id)
    elif method == "mmseq":
        with open(cluster_file, 'r') as file:
            cluster_ids = {}
            current_cluster_id = 1
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                cluster_key = parts[0]
                if cluster_key not in cluster_ids:
                    cluster_ids[cluster_key] = current_cluster_id
                    current_cluster_id += 1
                sequence_id = parts[1].split('|')[0]
                clusters[cluster_ids[cluster_key]].append(sequence_id)
    return clusters

def split_train_test_by_size(cluster_sizes, train_ratio, margin=0.05):
    """
    Partition clusters into train/test using the Karmarkar-Karp differencing
    algorithm for balanced number partitioning.
    
    The partition closer to (train_ratio - margin) * total_sequences
    is assigned to train. If test would be empty, the smallest train
    cluster is moved to test.
    """
    adjusted_train_ratio = train_ratio - margin
    total_sequences = sum(cluster_sizes.values())

    if len(cluster_sizes) <= 1:
        return list(cluster_sizes.keys()), []

    target_train_size = int(adjusted_train_ratio * total_sequences)

    heap = []
    counter = 0
    for cid, size in cluster_sizes.items():
        heapq.heappush(heap, (-size, counter, [cid], []))
        counter += 1

    while len(heap) > 1:
        neg_size1, _, a1, b1 = heapq.heappop(heap)
        neg_size2, _, a2, b2 = heapq.heappop(heap)
        new_size = neg_size1 - neg_size2
        new_a = a1 + b2
        new_b = b1 + a2
        heapq.heappush(heap, (new_size, counter, new_a, new_b))
        counter += 1

    _, _, partition_a, partition_b = heap[0]

    size_a = sum(cluster_sizes[c] for c in partition_a)
    size_b = sum(cluster_sizes[c] for c in partition_b)

    if abs(size_a - target_train_size) <= abs(size_b - target_train_size):
        train_clusters, test_clusters = partition_a, partition_b
    else:
        train_clusters, test_clusters = partition_b, partition_a

    if not test_clusters and len(train_clusters) > 1:
        smallest = min(train_clusters, key=lambda c: cluster_sizes[c])
        train_clusters.remove(smallest)
        test_clusters.append(smallest)

    return train_clusters, test_clusters

def build_sequence_index(fasta_file):
    """
    Parse a FASTA file once and return a dict mapping sequence_id → SeqRecord.
    """
    index = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        key = record.id.split('|')[0]
        index[key] = record
    return index


def write_sequences_to_fasta(fasta_file, clusters, output_file):
    """DEPRECATED: Use write_sequences_to_fasta_from_index instead."""
    logger.warning("write_sequences_to_fasta is deprecated — "
                   "use write_sequences_to_fasta_from_index for better performance")
    if os.path.abspath(fasta_file) == os.path.abspath(output_file):
        raise ValueError(
            "Input and output are the same path — would cause data corruption. "
            "Use a temp file + os.replace().")
    sequence_ids = {seq_id for cluster_id in clusters for seq_id in clusters[cluster_id]}
    with open(output_file, 'w') as out_fasta:
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id.split('|')[0] in sequence_ids:
                SeqIO.write(record, out_fasta, "fasta")


def write_sequences_to_fasta_from_index(seq_index, clusters, output_file):
    """
    Write sequences using the in-memory index instead of re-parsing disk.
    
    Parameters
    ----------
    seq_index : dict[str, SeqRecord]
    clusters  : dict[cluster_id, list[seq_id]]
    output_file : str
    """
    target_ids = {seq_id for cid in clusters for seq_id in clusters[cid]}
    records = [seq_index[sid] for sid in target_ids if sid in seq_index]
    SeqIO.write(records, output_file, "fasta")
    return len(records)

def create_diamond_db(train_fasta, db_name):
    run_command(
        f"diamond makedb --in {shlex.quote(train_fasta)}"
        f" -d {shlex.quote(db_name)}",
        description="DIAMOND makedb",
    )


def run_diamond_alignment(test_fasta, db_name, output_file,
                          identity_threshold, query_cover,
                          subject_cover=None):
    """
    If subject_cover is provided, add --subject-cover to match
    bidirectional coverage mode.
    """
    diamond_identity = identity_threshold * 100
    cmd = (
        f"diamond blastp"
        f" -d {shlex.quote(db_name)}"
        f" -q {shlex.quote(test_fasta)}"
        f" --id {diamond_identity}"
        f" --query-cover {query_cover}"
    )
    if subject_cover is not None:
        cmd += f" --subject-cover {subject_cover}"
    cmd += f" -o {shlex.quote(output_file)} --quiet"

    run_command(cmd, description="DIAMOND blastp")
    return os.path.exists(output_file) and os.path.getsize(output_file) != 0

def filter_diamond_output(diamond_output):
    qualifying_sequences, reference_sequences = set(), set()
    with open(diamond_output, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            qualifying_sequences.add(parts[0].split('|')[0])
            reference_sequences.add(parts[1].split('|')[0])
    return qualifying_sequences, reference_sequences

def iterative_diamond_alignment_and_reassignment(
        train_clusters, test_clusters, clusters,
        seq_index,
        label_dir, train_ratio, identity_threshold, query_cover,
        subject_cover=None):

    train_output_file = os.path.join(label_dir, "train.fasta")
    test_output_file  = os.path.join(label_dir, "test.fasta")

    write_sequences_to_fasta_from_index(
        seq_index, {cl: clusters[cl] for cl in train_clusters}, train_output_file)
    write_sequences_to_fasta_from_index(
        seq_index, {cl: clusters[cl] for cl in test_clusters}, test_output_file)

    while True:
        if os.path.getsize(train_output_file) == 0 or os.path.getsize(test_output_file) == 0:
            break

        db_name = os.path.join(label_dir, "train_db")
        diamond_output = os.path.join(label_dir, "diamond_matches.m8")
        create_diamond_db(train_output_file, db_name)

        success = run_diamond_alignment(
            test_output_file, db_name, diamond_output,
            identity_threshold, query_cover,
            subject_cover=subject_cover)
        if not success:
            break

        qualifying_sequences, reference_sequences = filter_diamond_output(diamond_output)
        if not qualifying_sequences:
            break

        clusters_to_reassign = {cluster_id for cluster_id in test_clusters if set(clusters[cluster_id]).intersection(qualifying_sequences)}
        new_test_clusters = set(test_clusters) - clusters_to_reassign
        new_train_clusters = set(train_clusters).union(clusters_to_reassign)

        total_sequences = sum(len(clusters[cl]) for cl in clusters)
        new_train_set_size = sum(len(clusters[cl]) for cl in new_train_clusters)

        if new_train_set_size / total_sequences > train_ratio or len(new_test_clusters) < 2:
            break

        train_clusters, test_clusters = list(new_train_clusters), list(new_test_clusters)
        write_sequences_to_fasta_from_index(
            seq_index, {cl: clusters[cl] for cl in train_clusters}, train_output_file)
        write_sequences_to_fasta_from_index(
            seq_index, {cl: clusters[cl] for cl in test_clusters}, test_output_file)

    return train_clusters, test_clusters



def reassign_leaking_test_to_train(train_file, test_file, output_dir,
                                   identity_threshold, query_cover,
                                   subject_cover=None):
    """
    Global cross-label DIAMOND check.
    Any TEST sequence that hits the TRAIN set above threshold is moved
    from test into train (not deleted from train).
    
    Iterates until no more leaking test sequences are found.
    
    Returns (leaking_test_ids: set) — the IDs moved from test to train.
    """
    all_moved = set()
    iteration = 0

    while True:
        iteration += 1
        db_name       = os.path.join(output_dir, "global_train_db")
        diamond_output = os.path.join(output_dir, "global_diamond_hits.m8")

        if (os.path.getsize(train_file) == 0 or
                os.path.getsize(test_file) == 0):
            break

        create_diamond_db(train_file, db_name)
        has_hits = run_diamond_alignment(
            test_file, db_name, diamond_output,
            identity_threshold, query_cover,
            subject_cover=subject_cover)
        if not has_hits:
            break

        leaking_test_ids, _ = filter_diamond_output(diamond_output)
        if not leaking_test_ids:
            break

        logger.info(
            f"Global DIAMOND iteration {iteration}: "
            f"moving {len(leaking_test_ids)} test sequences → train"
        )
        all_moved.update(leaking_test_ids)

        leaking_records = []
        clean_records   = []
        for record in SeqIO.parse(test_file, "fasta"):
            rid = record.id.split('|')[0]
            if rid in leaking_test_ids:
                leaking_records.append(record)
            else:
                clean_records.append(record)

        train_tmp = train_file + ".tmp"
        with open(train_tmp, 'w') as out:
            for record in SeqIO.parse(train_file, "fasta"):
                SeqIO.write(record, out, "fasta")
            for record in leaking_records:
                SeqIO.write(record, out, "fasta")
        os.replace(train_tmp, train_file)

        test_tmp = test_file + ".tmp"
        SeqIO.write(clean_records, test_tmp, "fasta")
        os.replace(test_tmp, test_file)

    return all_moved

def process_single_label(label, fasta_file, output_dir, temp_dir, args):
    """
    Process a single label: cluster → split → DIAMOND refinement.
    Returns (label, train_output, test_output, n_train, n_test).
    
    This function is self-contained and safe to run in a separate process.
    """
    label_dir   = os.path.join(output_dir, label)
    label_work_dir = os.path.join(temp_dir, f"{label}_work")
    os.makedirs(label_work_dir, exist_ok=True)
    
    output_file = os.path.join(label_work_dir, f"{label}_{args.method}")

    if args.method == "cdhit":
        run_cdhit(fasta_file, output_file, args.identity_threshold, args.cdhit_cores)
        cluster_file = f"{output_file}.clstr"
    elif args.method == "mmseq":
        tmp_dir = os.path.join(label_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        run_mmseqs2_clustering(fasta_file, output_file, tmp_dir,
                               args.identity_threshold,
                               coverage=args.coverage,
                               cov_mode=args.cov_mode)
        cluster_file = f"{output_file}.tsv"

    validate_file_exists(cluster_file, args.method)
    clusters = parse_clusters(args.method, cluster_file)

    if not clusters:
        logger.warning(f"No clusters for label '{label}' — skipping")
        return label, None, None, 0, 0

    seq_index     = build_sequence_index(fasta_file)
    cluster_sizes = {c: len(ids) for c, ids in clusters.items()}

    train_clusters, test_clusters = split_train_test_by_size(
        cluster_sizes, args.train_ratio, margin=args.train_ratio_margin)

    subject_cover = int(args.coverage * 100) if args.cov_mode == 0 else None

    final_train, final_test = iterative_diamond_alignment_and_reassignment(
        train_clusters, test_clusters, clusters, seq_index,
        label_work_dir, args.train_ratio, args.identity_threshold, args.query_cover,
        subject_cover=subject_cover)

    train_out = os.path.join(temp_dir, f"{label}_train.fasta")
    test_out  = os.path.join(temp_dir, f"{label}_test.fasta")

    n_train = write_sequences_to_fasta_from_index(
        seq_index, {cl: clusters[cl] for cl in final_train}, train_out)
    n_test = write_sequences_to_fasta_from_index(
        seq_index, {cl: clusters[cl] for cl in final_test}, test_out)

    return label, train_out, test_out, n_train, n_test


def run_pipeline(args):
    """
    Run the full Fast-Part pipeline programmatically.
    
    Parameters
    ----------
    args : argparse.Namespace
        Configuration object with all pipeline parameters.
    
    Returns
    -------
    dict
        Statistics dictionary containing:
        - train_file: path to final training set
        - test_file: path to final test set
        - summary_file: path to summary file
        - train_label_counts: dict mapping labels to train counts
        - test_label_counts: dict mapping labels to test counts
        - label_sequence_counts: dict mapping labels to total counts
        - moved_ids: set of sequence IDs moved from test to train
        - initial_count: total number of sequences at start
        - elapsed_time: execution time in seconds
    """
    random.seed(args.seed)

    if args.identity_threshold < 0.4:
        args.method = "mmseq"
    
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "Temp")
    os.makedirs(temp_dir, exist_ok=True)

    summary_file = os.path.join(args.output_dir, "summary.txt")
    
    with open(summary_file, 'w') as summary:
        summary.write("Configuration Summary:\n")
        summary.write(f"Clustering Method: {args.method}\n")
        summary.write(f"Identity Threshold: {args.identity_threshold}\n")
        summary.write(f"Cores: {args.cdhit_cores}\n")
        summary.write(f"Train Ratio (adjusted): {args.train_ratio - args.train_ratio_margin}\n")
        summary.write(f"Query Cover: {args.query_cover}\n")
        summary.write(f"Coverage: {args.coverage}\n")
        summary.write(f"Coverage Mode: {args.cov_mode}\n")
        summary.write(f"Random Seed: {args.seed}\n\n")

    sequences_by_label = parse_and_separate_fasta(args.fasta_file)
    initial_sequence_count = sum(len(seqs) for seqs in sequences_by_label.values())

    label_sequence_counts = {lbl: len(seqs) for lbl, seqs in sequences_by_label.items()}
    train_label_counts = defaultdict(int)
    test_label_counts = defaultdict(int)
    label_train_files = []
    label_test_files  = []

    label_tasks = list(write_sequences_by_label(sequences_by_label, args.output_dir))

    if args.parallel_labels > 1:
        with ProcessPoolExecutor(max_workers=args.parallel_labels) as executor:
            futures = {
                executor.submit(
                    process_single_label,
                    label, fasta_file, args.output_dir, temp_dir, args
                ): label
                for label, fasta_file in label_tasks
            }

            for future in as_completed(futures):
                submitted_label = futures[future]
                try:
                    returned_label, train_out, test_out, n_train, n_test = future.result()
                    if train_out is None:
                        continue
                    
                    train_label_counts[returned_label] = n_train
                    test_label_counts[returned_label]  = n_test
                    label_train_files.append(train_out)
                    label_test_files.append(test_out)
                except Exception as exc:
                    logger.error(f"Label '{submitted_label}' failed: {exc}")
    else:
        for label, fasta_file in label_tasks:
            try:
                returned_label, train_out, test_out, n_train, n_test = \
                    process_single_label(label, fasta_file, args.output_dir, temp_dir, args)
                if train_out is None:
                    continue
                train_label_counts[returned_label] = n_train
                test_label_counts[returned_label]  = n_test
                label_train_files.append(train_out)
                label_test_files.append(test_out)
            except Exception as exc:
                logger.error(f"Label '{label}' failed: {exc}")

    train_file = os.path.join(args.output_dir, "Train.fasta")
    test_file  = os.path.join(args.output_dir, "Test.fasta")

    with open(train_file, 'w') as out:
        for f in label_train_files:
            with open(f, 'r') as inp:
                shutil.copyfileobj(inp, out)

    with open(test_file, 'w') as out:
        for f in label_test_files:
            with open(f, 'r') as inp:
                shutil.copyfileobj(inp, out)

    subject_cover = int(args.coverage * 100) if args.cov_mode == 0 else None
    moved_ids = reassign_leaking_test_to_train(
        train_file, test_file, temp_dir,
        args.identity_threshold, args.query_cover,
        subject_cover=subject_cover)
    logger.info(f"Global cross-label reassignment: {len(moved_ids)} test→train")

    train_label_counts = defaultdict(int)
    test_label_counts  = defaultdict(int)
    for record in SeqIO.parse(train_file, "fasta"):
        lbl = sanitize_label(record.description.split('|')[1])
        train_label_counts[lbl] += 1
    for record in SeqIO.parse(test_file, "fasta"):
        lbl = sanitize_label(record.description.split('|')[1])
        test_label_counts[lbl] += 1

    all_labels = set(label_sequence_counts.keys())
    train_labels = set(train_label_counts.keys())
    test_labels = set(test_label_counts.keys())
    missing_train_labels = all_labels - train_labels
    missing_test_labels = all_labels - test_labels

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if item_path not in {train_file, test_file, summary_file}:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
    else:
        logger.info(f"Temporary files preserved in: {temp_dir}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open(summary_file, 'a') as summary:
        summary.write(f"Initial total sequence count: {initial_sequence_count}\n")
        summary.write("Sequence counts per label:\n")
        for label, count in label_sequence_counts.items():
            summary.write(f"  {label}: {count}\n")
        summary.write(f"\nFinal sequence counts per label in Train set:\n")
        for label, count in train_label_counts.items():
            total = label_sequence_counts.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            summary.write(f"  {label}: {count} ({percentage:.2f}%)\n")
        summary.write(f"\nFinal sequence counts per label in Test set:\n")
        for label, count in test_label_counts.items():
            total = label_sequence_counts.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            summary.write(f"  {label}: {count} ({percentage:.2f}%)\n")
        summary.write(f"\nTotal sequences reassigned (test→train): {len(moved_ids)}\n")
        summary.write("Reassigned sequence IDs:\n")
        for seq_id in moved_ids:
            summary.write(f"  {seq_id}\n")
        summary.write(f"\nNumber of labels in Train set: {len(train_labels)}\n")
        summary.write(f"Number of labels in Test set: {len(test_labels)}\n")
        summary.write("Missing labels in Train set:\n")
        for label in missing_train_labels:
            summary.write(f"  {label}\n")
        summary.write("Missing labels in Test set:\n")
        for label in missing_test_labels:
            summary.write(f"  {label}\n")
        summary.write(f"\nExecution time: {elapsed_time:.2f} seconds\n")

    return {
        "train_file": train_file,
        "test_file": test_file,
        "summary_file": summary_file,
        "train_label_counts": dict(train_label_counts),
        "test_label_counts": dict(test_label_counts),
        "label_sequence_counts": label_sequence_counts,
        "moved_ids": moved_ids,
        "initial_count": initial_sequence_count,
        "elapsed_time": elapsed_time,
    }


def main():
    """Command-line entry point for Fast-Part pipeline."""
    args = get_args()
    results = run_pipeline(args)
    
    print(f"Updated combined train file is saved to {results['train_file']}")
    print(f"Updated combined test file is saved to {results['test_file']}")
    print(f"Summary file is saved to {results['summary_file']}")
    print(f"Execution time: {results['elapsed_time']:.2f} seconds")

if __name__ == "__main__":
    main()
