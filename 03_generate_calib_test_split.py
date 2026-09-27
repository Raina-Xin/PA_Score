import os
import sys
import json
import random
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Tuple
from data_utils import load_dataset
from config import load_config, Config
import argparse


def normalize_fraction(f: float) -> str:
    """Return a normalized string like '0p80' for 0.8, up to 2 decimals."""
    return str(round(f, 2)).replace(".", "p")


def compute_checksum(id_list: List[str]) -> str:
    """Compute short md5 checksum of the sorted ID list."""
    md5 = hashlib.md5()
    for _id in sorted(id_list):
        md5.update(str(_id).encode("utf-8"))
    return md5.hexdigest()[:8]


def extract_ids(dataset: List[Dict], id_key: str) -> Tuple[List[str], int]:
    """
    Extract IDs by id_key. Warn and deduplicate duplicates, returning unique IDs and number removed.
    """
    ids: List[str] = []
    missing = 0
    for item in dataset:
        if id_key not in item:
            missing += 1
            continue
        ids.append(item[id_key])
    if missing > 0:
        logging.error(f"Missing '{id_key}' in {missing} items.")
        # We fail here since id_key missing implies we cannot proceed safely
        raise KeyError(f"Missing '{id_key}' in {missing} dataset items")

    # Deduplicate while preserving order of first occurrence
    seen = set()
    unique_ids: List[str] = []
    for _id in ids:
        if _id in seen:
            continue
        seen.add(_id)
        unique_ids.append(_id)
    duplicates_removed = len(ids) - len(unique_ids)
    if duplicates_removed > 0:
        logging.warning(f"Found {duplicates_removed} duplicate IDs; keeping first occurrence and removing the rest.")
    return unique_ids, duplicates_removed


def validate_fractions(train_fraction: float, calib_fraction: float, test_fraction: float) -> None:
    if not (0 < train_fraction < 1):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not (0 < calib_fraction < 1):
        raise ValueError(f"calib_fraction must be in (0, 1), got {calib_fraction}")
    if not (0 < test_fraction < 1):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    total = train_fraction + calib_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got train+calib+test = {total:.6f}"
        )


def maybe_stratified_split_three_way(
    ids: List[str],
    dataset: List[Dict],
    id_key: str,
    stratify_key: str,
    train_fraction: float,
    calib_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], bool]:
    """
    Best-effort stratified three-way split by stratify_key.
    Returns (train_ids, calib_ids, test_ids, used_stratify).
    Falls back to random if key missing or insufficient instances per stratum.
    """
    if not stratify_key:
        return [], [], [], False

    # Map id -> group
    id_to_group: Dict[str, str] = {}
    for item in dataset:
        if id_key in item and stratify_key in item:
            id_to_group[item[id_key]] = item[stratify_key]

    if len(id_to_group) < len(ids):
        logging.warning(
            f"Not all items have '{stratify_key}'; falling back to random split."
        )
        return [], [], [], False

    # Build groups
    from collections import defaultdict
    group_to_ids: Dict[str, List[str]] = defaultdict(list)
    for _id in ids:
        group_to_ids[id_to_group[_id]].append(_id)

    rng = random.Random(seed)
    train_ids: List[str] = []
    calib_ids: List[str] = []
    test_ids: List[str] = []

    # Require at least 3 items to make a three-way split in each group, otherwise fallback
    for group, group_ids in group_to_ids.items():
        if len(group_ids) < 3:
            logging.warning(
                f"Group '{group}' has {len(group_ids)} items; insufficient for stratified 3-way split. Falling back to random."
            )
            return [], [], [], False
        rng.shuffle(group_ids)
        n = len(group_ids)
        idx_train = int(n * train_fraction)
        idx_calib = int(n * (train_fraction + calib_fraction))
        train_ids.extend(group_ids[:idx_train])
        calib_ids.extend(group_ids[idx_train:idx_calib])
        test_ids.extend(group_ids[idx_calib:])

    return train_ids, calib_ids, test_ids, True


def split_calib_test(
    dataset: List[Dict],
    dataset_name: str,
    train_fraction: float = 0.5,
    calib_fraction: float = 0.3,
    test_fraction: float = 0.2,
    output_dir: Path = Path("calib_test_splits"),
    id_key: str = "id",
    seed: int = 42,
    debug: bool = False,
    stratify_key: str = "",
    dry_run: bool = False,
) -> int:
    """
    Randomly split the dataset into calibration and testing sets based on the given fraction,
    and save the question IDs to JSON files in the output directory.

    Examples
    --------
    python 03_generate_calib_test_split.py \
      --config_path path/to/config.yaml \
      --train_fraction 0.5 \
      --calib_fraction 0.3 \
      --test_fraction 0.2 \
      --seed 123 \
      --output_dir calib_test_splits \
      --id_key id \
      --debug
    """
    try:
        validate_fractions(train_fraction, calib_fraction, test_fraction)
    except Exception as e:
        logging.error(str(e))
        return 1

    if not dataset:
        logging.error("Dataset is empty.")
        return 1

    # Extract unique ids, warn & dedupe
    try:
        ids, duplicates_removed = extract_ids(dataset, id_key)
    except KeyError as e:
        logging.error(str(e))
        return 1

    if not ids:
        logging.error("No IDs extracted; cannot proceed.")
        return 1

    # Reproducibility
    random.seed(seed)

    # Maybe stratify (three-way)
    train_ids: List[str]
    calib_ids: List[str]
    test_ids: List[str]
    if stratify_key:
        tids, cids, vids, used = maybe_stratified_split_three_way(
            ids, dataset, id_key, stratify_key, train_fraction, calib_fraction, test_fraction, seed
        )
        if used:
            train_ids, calib_ids, test_ids = tids, cids, vids
            logging.info(f"Performed stratified three-way split by '{stratify_key}'.")
        else:
            # Fall back to random three-way
            shuffled = ids[:]
            random.shuffle(shuffled)
            n = len(shuffled)
            idx_train = int(n * train_fraction)
            idx_calib = int(n * (train_fraction + calib_fraction))
            train_ids = shuffled[:idx_train]
            calib_ids = shuffled[idx_train:idx_calib]
            test_ids = shuffled[idx_calib:]
            logging.info("Fell back to random three-way split.")
    else:
        shuffled = ids[:]
        random.shuffle(shuffled)
        n = len(shuffled)
        idx_train = int(n * train_fraction)
        idx_calib = int(n * (train_fraction + calib_fraction))
        train_ids = shuffled[:idx_train]
        calib_ids = shuffled[idx_train:idx_calib]
        test_ids = shuffled[idx_calib:]

    # Sanity checks
    n_unique = len(ids)
    if len(train_ids) + len(calib_ids) + len(test_ids) != n_unique:
        logging.error(
            f"Size mismatch: train({len(train_ids)}) + calib({len(calib_ids)}) + test({len(test_ids)}) != unique({n_unique})."
        )
        return 1
    overlap = set(train_ids).intersection(calib_ids) | set(train_ids).intersection(test_ids) | set(calib_ids).intersection(test_ids)
    if overlap:
        logging.error(f"Found overlap between splits: {len(overlap)} items.")
        return 1

    # Paths and filenames
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tr_norm = normalize_fraction(train_fraction)
    cf_norm = normalize_fraction(calib_fraction)
    tf_norm = normalize_fraction(test_fraction)

    train_file = output_dir / f"train_ids_{dataset_name}_tr{tr_norm}_cf{cf_norm}_tf{tf_norm}.json"
    calib_file = output_dir / f"calib_ids_{dataset_name}_tr{tr_norm}_cf{cf_norm}_tf{tf_norm}.json"
    test_file = output_dir / f"test_ids_{dataset_name}_tr{tr_norm}_cf{cf_norm}_tf{tf_norm}.json"
    manifest_file = output_dir / f"split_manifest_{dataset_name}_tr{tr_norm}_cf{cf_norm}_tf{tf_norm}.json"
    debug_file = output_dir / f"debug_summary_{dataset_name}_tr{tr_norm}_cf{cf_norm}_tf{tf_norm}.txt"

    # Checksums
    train_checksum = compute_checksum(train_ids)
    calib_checksum = compute_checksum(calib_ids)
    test_checksum = compute_checksum(test_ids)

    # Debug prints
    if debug:
        logging.debug(
            f"Total(unique): {n_unique} | calib: {len(calib_ids)} | test: {len(test_ids)} | overlap: 0"
        )
        logging.debug(f"First 5 calib IDs: {calib_ids[:5]}")
        logging.debug(f"First 5 test IDs: {test_ids[:5]}")

    # Optionally only report
    if dry_run:
        logging.info("Dry run: not writing any files.")
        return 0

    # Write splits
    with train_file.open("w") as f:
        json.dump(train_ids, f, indent=2)
    with calib_file.open("w") as f:
        json.dump(calib_ids, f, indent=2)
    with test_file.open("w") as f:
        json.dump(test_ids, f, indent=2)

    logging.info(f"Saved {len(train_ids)} training IDs to {train_file}")
    logging.info(f"Saved {len(test_ids)} testing IDs to {test_file}")
    logging.info(f"Saved {len(calib_ids)} calibration IDs to {calib_file}")

    # Manifest
    manifest = {
        "dataset_name": dataset_name,
        "train_fraction": train_fraction,
        "calib_fraction": calib_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "n_total": n_unique,
        "n_train": len(train_ids),
        "n_calib": len(calib_ids),
        "n_test": len(test_ids),
        "id_key": id_key,
        "duplicates_removed": duplicates_removed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "split_files": {
            "train": str(train_file),
            "calib": str(calib_file),
            "test": str(test_file),
        },
        "checksums": {
            "train": train_checksum,
            "calib": calib_checksum,
            "test": test_checksum,
        },
    }
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Wrote manifest to {manifest_file}")

    # Debug file
    if debug:
        lines = [
            f"Dataset: {dataset_name}",
            f"Seed: {seed}",
            f"ID key: {id_key}",
            f"Train fraction: {train_fraction} (tr{tr_norm})",
            f"Calib fraction: {calib_fraction} (cf{cf_norm})",
            f"Test fraction: {test_fraction} (tf{tf_norm})",
            f"Total unique: {n_unique}",
            f"Training: {len(train_ids)}",
            f"Calibration: {len(calib_ids)}",
            f"Test: {len(test_ids)}",
            f"Overlap: 0",
            f"Duplicates removed: {duplicates_removed}",
            f"Train checksum: {train_checksum}",
            f"Calib checksum: {calib_checksum}",
            f"Test checksum: {test_checksum}",
            f"First 5 train: {train_ids[:5]}",
            f"First 5 calib: {calib_ids[:5]}",
            f"First 5 test: {test_ids[:5]}",
        ]
        debug_file.write_text("\n".join(lines))
        logging.info(f"Wrote debug summary to {debug_file}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate calibration and test split IDs for a dataset."
    )
    parser.add_argument("--config_path", required=True, help="Path to config file.")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.5,
        required=False,
        help="Fraction of examples assigned to the training split (0<frac<1).",
    )
    parser.add_argument(
        "--calib_fraction",
        type=float,
        default=0.3,
        required=False,
        help="Fraction of examples assigned to the calibration split (0<frac<1).",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        required=False,
        help="Fraction of examples assigned to the test split (0<frac<1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="calib_test_splits",
        help="Directory to save split files.",
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="Key to use to extract IDs from dataset entries.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and write a debug summary file.",
    )
    parser.add_argument(
        "--stratify_key",
        type=str,
        default="",
        help="Optional key for stratified split; falls back to random if not feasible.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and log stats only; do not write files.",
    )

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    config: Config = load_config(args.config_path)
    dataset = load_dataset(config)
    dataset_name = getattr(config, "dataset", "dataset")

    exit_code = split_calib_test(
        dataset=dataset,
        dataset_name=dataset_name,
        train_fraction=args.train_fraction,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        output_dir=Path(args.output_dir),
        id_key=args.id_key,
        seed=args.seed,
        debug=args.debug,
        stratify_key=args.stratify_key,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)
