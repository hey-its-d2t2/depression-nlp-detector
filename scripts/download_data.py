#!/usr/bin/env python
"""Download and prepare public datasets for training.

Supported datasets:
    1. depression_reddit_cleaned (HuggingFace)
    2. synthetic (no download required)

Usage:
    python scripts/download_data.py --dataset synthetic --n-samples 5000
    python scripts/download_data.py --dataset huggingface --hf-name depression_reddit_cleaned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depression_detector.data.data_loader import DataLoader
from depression_detector.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/prepare training dataset")
    p.add_argument("--dataset", choices=["synthetic", "huggingface"], default="synthetic")
    p.add_argument("--n-samples", type=int, default=5000, help="Synthetic: number of samples")
    p.add_argument("--hf-name", default="depression_reddit_cleaned", help="HuggingFace dataset name")
    p.add_argument("--hf-token", default=None, help="HuggingFace API token")
    p.add_argument("--output", type=Path, default=Path("data/raw/dataset.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger("INFO", "logs/download.log")
    logger = get_logger("download_data")

    loader = DataLoader()

    if args.dataset == "synthetic":
        logger.info(f"Generating {args.n_samples} synthetic samples")
        dataset = loader.synthetic(n_samples=args.n_samples)
    else:
        logger.info(f"Downloading {args.hf_name} from HuggingFace Hub")
        dataset = loader.from_huggingface(args.hf_name, hf_token=args.hf_token)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_csv(args.output)
    logger.info(f"Dataset saved to {args.output}")
    logger.info(f"Class distribution: {dataset.class_distribution()}")
    logger.info(f"Total samples: {len(dataset)}")


if __name__ == "__main__":
    main()
