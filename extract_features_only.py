#!/usr/bin/env python
"""Feature extraction script - extracts and caches all multimodal features.

Runs the data pipeline with n_epochs=0 (no training) to trigger feature
extraction and caching for LLAMA3.2-3B (text), VJEPA2 (video), and
Wav2VecBert (audio). Cached features are reused by subsequent training runs.

Usage:
    python extract_features_only.py
"""

from exca import ConfDict
from algonauts2025.main import Experiment
from algonauts2025.grids.defaults import default_config

# Feature extraction config — no training, just cache features
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
        # Uncomment to test on a small subset first:
        # "study": {"query": "subject_timeline_index<50"},
    },
}


def extract_features(config: dict) -> None:
    """Extract and cache all features by building data loaders."""
    print("=" * 80)
    print("FEATURE EXTRACTION")
    print("=" * 80)

    task = Experiment(**config)

    print(f"Cache directory: {config['data']['text_feature']['infra']['folder']}")
    print(f"Dataset path: {config['data']['study']['path']}")
    print("=" * 80)

    print("\nExtracting features (this may take several hours)...")
    loaders = task.data.get_loaders(split_to_build=["train", "val", "test"])

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nCached features in: {config['data']['text_feature']['infra']['folder']}")
    for split, loader in loaders.items():
        print(f"  {split}: {len(loader)} batches")


if __name__ == "__main__":
    config = ConfDict(default_config)
    config.update(update)
    extract_features(config)
