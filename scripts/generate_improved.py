"""Generate Lee-filtered and CLAHE preprocessing variants for evaluation."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("SARDET_DATA_ROOT"),
        help="SARDet-100K root directory (or set SARDET_DATA_ROOT).",
    )
    parser.add_argument("--looks", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--lee-window-size", type=int, default=7)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-grid-size", type=int, default=8)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("lee", "clahe"),
        default=("lee", "clahe"),
    )
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root or SARDET_DATA_ROOT is required")
    if args.lee_window_size <= 0 or args.lee_window_size % 2 == 0:
        parser.error("--lee-window-size must be a positive odd integer")
    if args.clahe_grid_size <= 0:
        parser.error("--clahe-grid-size must be positive")
    return args


def lee_filter(channel, window_size=7):
    """Apply the local-statistics Lee filter used in the experiment."""
    values = channel.astype(np.float64)
    local_mean = uniform_filter(values, window_size)
    local_mean_sq = uniform_filter(values**2, window_size)
    local_variance = local_mean_sq - local_mean**2
    noise_variance = np.mean(
        local_variance / (local_mean**2 + 1e-10)
    )
    weight = local_variance / (
        local_variance + noise_variance * local_mean**2 + 1e-10
    )
    return np.clip(local_mean + weight * (values - local_mean), 0, 255)


def apply_lee(image, window_size=7):
    output = np.empty_like(image, dtype=np.float64)
    for channel_index in range(image.shape[2]):
        output[:, :, channel_index] = lee_filter(
            image[:, :, channel_index],
            window_size,
        )
    return output.astype(np.uint8)


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size,
    )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_directory(source_dir, destination_dir, transform):
    destination_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(source_dir.glob("*.*"))
    written = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        output_path = destination_dir / image_path.name
        if not cv2.imwrite(str(output_path), transform(image)):
            raise OSError(f"OpenCV could not write {output_path}")
        written += 1
    return written


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    methods = set(args.methods)

    if "lee" in methods:
        for looks in args.looks:
            source_dir = data_root / "JPEGImages" / f"test_speckle_L{looks}"
            destination_dir = data_root / "JPEGImages" / f"test_lee_L{looks}"
            written = process_directory(
                source_dir,
                destination_dir,
                lambda image: apply_lee(image, args.lee_window_size),
            )
            print(f"[lee] L={looks}: wrote {written} images to {destination_dir}")

    if "clahe" in methods:
        source_dir = data_root / "JPEGImages" / "test"
        destination_dir = data_root / "JPEGImages" / "test_clahe"
        grid_size = (args.clahe_grid_size, args.clahe_grid_size)
        written = process_directory(
            source_dir,
            destination_dir,
            lambda image: apply_clahe(
                image,
                clip_limit=args.clahe_clip_limit,
                grid_size=grid_size,
            ),
        )
        print(f"[clahe] Wrote {written} images to {destination_dir}")


if __name__ == "__main__":
    main()
