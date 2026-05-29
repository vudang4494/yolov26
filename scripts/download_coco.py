#!/usr/bin/env python3
"""
YOLOv26 Download COCO Dataset from HuggingFace.
"""
import argparse
import io
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys_path_insert = __import__("sys").path.insert
sys_path_insert(0, str(Path(__file__).parent.parent))


def download_coco(split="val", limit=0, output_dir="dataset/coco"):
    """
    Download COCO dataset from HuggingFace detection-datasets.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        os.system(f"{__import__('sys').executable} -m pip install datasets -q")
        from datasets import load_dataset

    from PIL import Image

    output_path = Path(output_dir)
    images_dir = output_path / "images" / split
    labels_dir = output_path / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading COCO2017 {split} from HuggingFace...")
    dataset = load_dataset("detection-datasets/coco", split=split, streaming=True)
    dataset_iter = iter(dataset)

    count = 0
    downloaded = 0

    print(f"Processing up to {limit if limit > 0 else 'all'} images...")

    for item in tqdm(dataset_iter, total=limit if limit > 0 else None):
        if limit > 0 and count >= limit:
            break

        img_id = item["image_id"]
        img_bytes = item["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes))
        img_path = images_dir / f"{img_id:012d}.jpg"
        img.save(img_path, "JPEG")
        downloaded += 1

        if "objects" in item:
            bbox_labels = item["objects"]
            h, w = item.get("height", img.height), item.get("width", img.width)
            label_lines = []

            for box, label in zip(bbox_labels["bbox"], bbox_labels["label"]):
                x, y, bw, bh = box
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                nw = max(0.001, min(1, nw))
                nh = max(0.001, min(1, nh))
                label_lines.append(f"{int(label)} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

            if label_lines:
                label_path = labels_dir / f"{img_id:012d}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

        count += 1

    print(f"\nDownloaded {downloaded} images to {output_path}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download COCO Dataset")
    parser.add_argument("--split", type=str, default="val",
                       help="Split: train or val")
    parser.add_argument("--limit", type=int, default=5000,
                       help="Limit number of images (0 = all)")
    parser.add_argument("--output-dir", type=str, default="dataset/coco",
                       help="Output directory")
    args = parser.parse_args()

    download_coco(args.split, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
