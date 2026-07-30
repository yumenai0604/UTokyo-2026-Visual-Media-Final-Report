"""Prepare the robustness-analysis subsets used in the final report.

The script can generate multiplicative Gamma speckle images, a COCO subset
containing images with small objects, and source-specific COCO subsets.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_PATTERNS = ("*.jpg", "*.png", "*.bmp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("SARDET_DATA_ROOT"),
        help="SARDet-100K root directory (or set SARDET_DATA_ROOT).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("speckle", "small", "source"),
        default=("speckle", "small", "source"),
        help="Artifacts to generate.",
    )
    parser.add_argument(
        "--looks",
        nargs="+",
        type=int,
        default=(1, 2, 4),
        help="Equivalent numbers of looks for the Gamma noise model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--small-area",
        type=float,
        default=32**2,
        help="COCO area threshold in pixels for a small object.",
    )
    parser.add_argument(
        "--min-source-images",
        type=int,
        default=20,
        help="Skip source subsets with fewer images than this value.",
    )
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root or SARDET_DATA_ROOT is required")
    return args


def add_speckle(image, looks):
    """Apply the multiplicative SAR speckle model used in the report."""
    if looks <= 0:
        raise ValueError("looks must be positive")
    noise_2d = np.random.gamma(
        shape=looks,
        scale=1.0 / looks,
        size=image.shape[:2],
    )
    noise = noise_2d[:, :, np.newaxis]
    return np.clip(image.astype(np.float64) * noise, 0, 255).astype(np.uint8)


def image_files(directory):
    # Keep the original experiment's extension-grouped iteration order.
    return [path for pattern in IMAGE_PATTERNS for path in directory.glob(pattern)]


def generate_speckle_sets(data_root, looks_values, seed):
    input_dir = data_root / "JPEGImages" / "test"
    files = image_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No test images found in {input_dir}")

    np.random.seed(seed)
    print(f"[speckle] Found {len(files)} test images")
    for looks in looks_values:
        output_dir = data_root / "JPEGImages" / f"test_speckle_L{looks}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in files:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"OpenCV could not read {image_path}")
            output_path = output_dir / image_path.name
            if not cv2.imwrite(str(output_path), add_speckle(image, looks)):
                raise OSError(f"OpenCV could not write {output_path}")
        print(f"[speckle] L={looks}: wrote {len(files)} images to {output_dir}")


def load_coco(annotation_file):
    with annotation_file.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_coco(path, coco, images, annotations):
    payload = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": images,
        "annotations": annotations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def generate_small_object_subset(data_root, coco, area_threshold):
    small_image_ids = {
        annotation["image_id"]
        for annotation in coco["annotations"]
        if annotation.get(
            "area",
            annotation["bbox"][2] * annotation["bbox"][3],
        )
        < area_threshold
    }
    images = [image for image in coco["images"] if image["id"] in small_image_ids]
    annotations = [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] in small_image_ids
    ]
    output_path = data_root / "Annotations" / "subsets" / "test_small_obj.json"
    write_coco(output_path, coco, images, annotations)
    print(
        f"[small] Wrote {output_path} "
        f"({len(images)} images, {len(annotations)} annotations)"
    )


def generate_source_subsets(data_root, coco, minimum_images):
    source_groups = defaultdict(list)
    for image in coco["images"]:
        token = image["file_name"].split("_")[0]
        prefix = "".join(character for character in token if not character.isdigit())
        source_groups[prefix].append(image["id"])

    output_dir = data_root / "Annotations" / "subsets"
    generated = 0
    for source, image_ids in source_groups.items():
        if len(image_ids) < minimum_images:
            continue
        image_id_set = set(image_ids)
        images = [image for image in coco["images"] if image["id"] in image_id_set]
        annotations = [
            annotation
            for annotation in coco["annotations"]
            if annotation["image_id"] in image_id_set
        ]
        output_path = output_dir / f"test_source_{source}.json"
        write_coco(output_path, coco, images, annotations)
        generated += 1
        print(
            f"[source] Wrote {output_path} "
            f"({len(images)} images, {len(annotations)} annotations)"
        )
    print(f"[source] Generated {generated} source subsets")


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    tasks = set(args.tasks)

    if "speckle" in tasks:
        generate_speckle_sets(data_root, args.looks, args.seed)

    if tasks.intersection(("small", "source")):
        annotation_file = data_root / "Annotations" / "test.json"
        coco = load_coco(annotation_file)
        if "small" in tasks:
            generate_small_object_subset(data_root, coco, args.small_area)
        if "source" in tasks:
            generate_source_subsets(data_root, coco, args.min_source_images)


if __name__ == "__main__":
    main()
