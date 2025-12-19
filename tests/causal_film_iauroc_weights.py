import argparse
import os
import sys

import numpy as np


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


sys.path.insert(0, _project_root())

from src.dataset import WeldingDataset
from configs.dataset_config import (
    DATA_ROOT,
    MANIFEST_PATH,
    VIDEO_LENGTH,
    AUDIO_SAMPLE_RATE,
    AUDIO_DURATION,
    SENSOR_LENGTH,
    IMAGE_SIZE,
    IMAGE_NUM_ANGLES,
)


ORDERED_CLASSES = [
    "Good",
    "Excessive_Convexity",
    "Undercut",
    "Lack_of_Fusion",
    "Porosity",
    "Spatter",
    "Burnthrough",
    "Porosity_w_Excessive_Penetration",
    "Excessive_Penetration",
    "Crater_Cracks",
    "Warping",
    "Overlap",
]

LABEL_TO_DISPLAY = {
    0: "Good",
    1: "Excessive_Convexity",
    2: "Undercut",
    3: "Lack_of_Fusion",
    4: "Porosity_w_Excessive_Penetration",
    5: "Porosity",
    6: "Spatter",
    7: "Burnthrough",
    8: "Excessive_Penetration",
    9: "Crater_Cracks",
    10: "Warping",
    11: "Overlap",
}


def compute_weights(dataset: WeldingDataset):
    labels = np.asarray(getattr(dataset, "_labels", []), dtype=int)

    if labels.size == 0:
        raise RuntimeError("Dataset labels are empty. Check manifest and data_root.")

    total = int(labels.size)
    neg_count = int((labels == 0).sum())
    pos_total = int(total - neg_count)

    counts = {name: 0 for name in ORDERED_CLASSES}
    for y in labels.tolist():
        name = LABEL_TO_DISPLAY.get(int(y), "unknown")
        if name in counts:
            counts[name] += 1

    rows = []
    for name in ORDERED_CLASSES:
        if name == "Good":
            rows.append(
                {
                    "name": name,
                    "count": counts[name],
                    "sample_ratio": counts[name] / total,
                    "pos_pair_weight": None,
                }
            )
            continue

        c = counts[name]
        pos_pair_weight = (c / pos_total) if pos_total > 0 else 0.0
        rows.append(
            {
                "name": name,
                "count": c,
                "sample_ratio": c / total,
                "pos_pair_weight": pos_pair_weight,
            }
        )

    return {
        "total": total,
        "neg_count": neg_count,
        "pos_total": pos_total,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Print per-defect-type weights implied by overall I-AUROC on Causal-FiLM test split"
    )
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--manifest_path", type=str, default=MANIFEST_PATH)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset = WeldingDataset(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        split="test",
        video_length=VIDEO_LENGTH,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        sensor_length=SENSOR_LENGTH,
        image_size=IMAGE_SIZE,
        num_angles=IMAGE_NUM_ANGLES,
        dummy=False,
        augment=False,
    )

    stats = compute_weights(dataset)

    print("=" * 60)
    print("Causal-FiLM test split: overall I-AUROC per-defect weights")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Good (negative) samples: {stats['neg_count']}")
    print(f"Defect (positive) samples: {stats['pos_total']}")
    print("=")
    print("pos_pair_weight: each defect type's fraction among ALL defect samples")
    print("(equivalently, its share of positive-negative pairs in AUROC)")
    print("-")

    for r in stats["rows"]:
        name = r["name"]
        if name == "Good":
            print(f"  {name:40s}: N/A (count={r['count']}, sample_ratio={r['sample_ratio']:.6f})")
        else:
            print(
                f"  {name:40s}: weight={r['pos_pair_weight']:.6f} "
                f"(count={r['count']}, sample_ratio={r['sample_ratio']:.6f})"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
