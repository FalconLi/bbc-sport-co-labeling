"""Export the BBCSport protocol to the general NPZ train/evaluate interface."""

import argparse
from pathlib import Path

from colabeling.data import ROOT, load_data, make_split
from colabeling.io import save_dataset


def main():
    parser = argparse.ArgumentParser(description="Export one BBCSport split as train.npz and test.npz.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/bbcsport-npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset = load_data()
    split = make_split(dataset, args.seed)
    save_dataset(args.output / "train.npz", split.train_views, y=split.known_classes,
                 known_mask=split.known_mask, sample_ids=split.train)
    save_dataset(args.output / "test.npz", split.test_views, y=dataset.labels[split.test], sample_ids=split.test)
    print(f"Saved train.npz (10 supplied labels + 360 unknown) and test.npz (367 articles) in {args.output}")


if __name__ == "__main__":
    main()
