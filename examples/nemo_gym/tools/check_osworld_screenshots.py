#!/usr/bin/env python3
"""Detect black or nearly blank screenshots in OSWorld rollout traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageStat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("--max-mean", type=float, default=8.0)
    parser.add_argument("--max-stddev", type=float, default=4.0)
    parser.add_argument("--min-files", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screenshots = sorted(args.trajectory_dir.glob("**/step_*.png"))
    results = []
    suspicious = []

    for path in screenshots:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
            mean = float(stat.mean[0])
            stddev = float(stat.stddev[0])
            extrema = grayscale.getextrema()

        row = {
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "mean_luma": round(mean, 3),
            "stddev_luma": round(stddev, 3),
            "min_luma": extrema[0],
            "max_luma": extrema[1],
        }
        results.append(row)
        if mean <= args.max_mean and stddev <= args.max_stddev:
            suspicious.append(row)

    summary = {
        "trajectory_dir": str(args.trajectory_dir),
        "screenshots": len(screenshots),
        "suspicious": len(suspicious),
        "suspicious_files": suspicious,
        "latest": results[-5:],
    }
    print(json.dumps(summary, indent=2))

    if len(screenshots) < args.min_files:
        raise SystemExit(2)
    if suspicious:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
