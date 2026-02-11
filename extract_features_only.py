#!/usr/bin/env python -u
"""Feature extraction script with step-by-step progress logging.

Extracts and caches multimodal features (LLAMA3.2, VJEPA2, Wav2VecBert).
Writes progress to a log file directly (no pipe buffering issues).

Usage:
    python -u extract_features_only.py [--logfile /path/to/log]
"""

import argparse
import logging
import os
import sys
import time

# Force unbuffered output everywhere
os.environ["PYTHONUNBUFFERED"] = "1"


def setup_logging(logfile=None):
    """Set up logging to both console and a log file, all flushed immediately."""
    handlers = []

    # Console handler (always)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    handlers.append(console)

    # File handler (if logfile specified)
    if logfile:
        fh = logging.FileHandler(logfile, mode="w")
        fh.setLevel(logging.INFO)
        handlers.append(fh)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    # Also redirect tqdm (stderr) to the log file
    if logfile:
        # Open the file in line-buffered mode for tqdm
        tqdm_file = open(logfile, "a", buffering=1)
        sys.stderr = tqdm_file

    return logging.getLogger("feature_extraction")


def elapsed(t0):
    m, s = divmod(time.time() - t0, 60)
    return f"{int(m)}m{int(s)}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        default=os.environ.get("FEAT_LOGFILE", None),
        help="Path to log file (also set via FEAT_LOGFILE env var)",
    )
    args = parser.parse_args()

    logger = setup_logging(args.logfile)
    t_start = time.time()

    # --- Step 1: Import and configure ---
    logger.info("Step 1/4: Importing modules...")
    t = time.time()
    from exca import ConfDict

    from algonauts2025.grids.defaults import default_config
    from algonauts2025.main import Experiment

    update = {
        "n_epochs": 0,
        "infra": {
            "cluster": "local",
            "gpus_per_node": 1,
            "mem_gb": 64,
            "timeout_min": 720,
        },
        "data": {
            "num_workers": 4,
            "study": {"query": "subject_timeline_index<10"},
        },
    }

    config = ConfDict(default_config)
    config.update(update)
    logger.info(f"  Done ({elapsed(t)})")
    logger.info(f"  Cache: {config['data']['text_feature']['infra']['folder']}")
    logger.info(f"  Dataset: {config['data']['study']['path']}")

    # --- Step 2: Initialize experiment (loads study, runs enhancers) ---
    logger.info("Step 2/4: Initializing experiment (study loading + enhancers)...")
    t = time.time()
    task = Experiment(**config)
    logger.info(f"  Done ({elapsed(t)})")

    # --- Step 3: Build data loaders (triggers feature extraction) ---
    logger.info("Step 3/4: Building data loaders and extracting features...")
    logger.info("  This triggers LLAMA3.2, VJEPA2, and Wav2VecBert extraction.")
    t = time.time()
    loaders = task.data.get_loaders(split_to_build=["train", "val", "test"])
    logger.info(f"  Done ({elapsed(t)})")

    # --- Step 4: Summary ---
    logger.info("=" * 60)
    logger.info(f"FEATURE EXTRACTION COMPLETE ({elapsed(t_start)} total)")
    logger.info("=" * 60)
    logger.info(f"Cache: {config['data']['text_feature']['infra']['folder']}")
    for split, loader in loaders.items():
        logger.info(f"  {split}: {len(loader)} batches")


if __name__ == "__main__":
    main()
