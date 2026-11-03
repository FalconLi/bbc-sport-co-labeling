"""Small synthetic, three-view SSL/MIL/ROD datasets for exercising the API."""

import argparse
from pathlib import Path

import numpy as np

from colabeling.data import ROOT
from colabeling.io import save_dataset


def make_examples(output, seed=42):
    output = Path(output)
    rng = np.random.default_rng(seed)
    # SSL has three named classes and three independently perturbed views.
    classes = np.array(["blue", "green", "red"])
    for split, per_class in [("train", 15), ("test", 8)]:
        labels = np.repeat(np.arange(3), per_class)
        views = [np.eye(3)[labels] * 3 + rng.normal(0, .25, (len(labels), 3)) for _ in range(3)]
        arrays = {"y": classes[labels]}
        if split == "train":
            arrays["known_mask"] = np.tile(np.arange(per_class) < 2, 3)
        save_dataset(output / "ssl" / f"{split}.npz", views, **arrays)
    # MIL labels belong to bags. Positive bags have one hidden positive instance.
    for split, count in [("train", 12), ("test", 8)]:
        bags = np.repeat(np.arange(count), 4)
        bag_labels = np.where(np.arange(count) % 2, "positive", "negative")
        repeated = np.repeat(bag_labels, 4)
        x = rng.normal(-2, .2, (count * 4, 3))
        x[(repeated == "positive") & (np.arange(len(x)) % 4 == 0)] += 4
        views = [x + rng.normal(0, .1, x.shape) for _ in range(3)]
        save_dataset(output / "mil" / f"{split}.npz", views, y=repeated, bag_ids=bags)
    # ROD training contains normal references and an unlabeled 25%-outlier pool.
    for split, normal_count, outlier_count in [("train", 24, 6), ("test", 18, 6)]:
        x = np.r_[rng.normal(2, .2, (normal_count, 3)), rng.normal(-2, .2, (outlier_count, 3))]
        views = [x + rng.normal(0, .1, x.shape) for _ in range(3)]
        arrays = ({"known_normal_mask": np.arange(len(x)) < 6} if split == "train" else
                  {"y": np.array(["normal"] * normal_count + ["outlier"] * outlier_count)})
        save_dataset(output / "rod" / f"{split}.npz", views, **arrays)
    return output


def main():
    parser = argparse.ArgumentParser(description="Create small synthetic examples for all three supported tasks.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_examples(args.output, args.seed)
    print(f"Saved SSL, MIL and ROD train/test pairs in {args.output}. These are synthetic checks, not paper benchmarks.")


if __name__ == "__main__":
    main()
