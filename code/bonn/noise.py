#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch add Gaussian noise to EEG txt files.

Input:
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data
Output:
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data10
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data20
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data30
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data40
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data50
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data60
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data70
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data80
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data90
    /mnt/gs21/scratch/jiangj33/ldx/case1/eeg_data100

Noise definition:
    noisy = x + N(0, (ratio * std(x))^2)

This keeps:
    - same number of txt files
    - same length per txt file
    - original labels / filenames
    - same file suffix case (.txt or .TXT)
    - reproducible results with fixed random seed
"""

from pathlib import Path
import numpy as np


# 根目录定义，方便后续统一修改
BASE_DIR = Path("/mnt/gs21/scratch/jiangj33/ldx/case17")
INPUT_DIR = BASE_DIR / "eeg_data"

# 噪声比例列表：10% 到 100%
NOISE_RATIOS = [i / 100 for i in range(10, 101, 10)]

# 自动生成输出目录
OUTPUT_DIRS = {
    ratio: BASE_DIR / f"eeg_data{int(ratio * 100)}"
    for ratio in NOISE_RATIOS
}


def add_noise(signal: np.ndarray, noise_ratio: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)

    # If a file is constant, use 1.0 to avoid zero noise scale.
    scale = float(np.std(signal))
    if scale == 0.0:
        scale = 1.0

    noise = rng.normal(loc=0.0, scale=noise_ratio * scale, size=signal.shape)
    return signal + noise


def process_dataset(input_dir: Path, output_dir: Path, noise_ratio: float, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 同时匹配 .txt 和 .TXT 文件（Linux 系统大小写敏感）
    txt_files = sorted(input_dir.glob("*.txt")) + sorted(input_dir.glob("*.TXT"))

    # 可选：若需统一排序（不分大小写），可取消下面注释
    # txt_files = sorted(txt_files, key=lambda f: f.name.lower())

    if not txt_files:
        raise FileNotFoundError(f"No .txt or .TXT files found in: {input_dir}")

    rng = np.random.default_rng(seed)

    for file_path in txt_files:
        x = np.loadtxt(file_path).reshape(-1)
        x_noisy = add_noise(x, noise_ratio=noise_ratio, rng=rng)

        # 保留原始文件名（包括后缀的大小写）
        out_path = output_dir / file_path.name
        np.savetxt(out_path, x_noisy, fmt="%.10f")

    print(f"Done: {len(txt_files)} files written to {output_dir}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    for ratio, out_dir in OUTPUT_DIRS.items():
        process_dataset(INPUT_DIR, out_dir, ratio, seed=42)

    print("\nAll noisy datasets generated successfully!")
    print(f"Input data: {INPUT_DIR}")
    print("Output datasets:")
    for ratio, out_dir in OUTPUT_DIRS.items():
        print(f"  - {int(ratio*100)}% noise: {out_dir}")


if __name__ == "__main__":
    main()