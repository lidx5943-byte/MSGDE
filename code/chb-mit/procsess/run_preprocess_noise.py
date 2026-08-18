#!/usr/bin/env python3
"""
加入高斯白噪声的鲁棒性测试脚本
对每个噪声比例生成样本间相似矩阵
"""

import sys
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.chbmit_loader import load_chbmit_subject, extract_segments
from src.preprocessing.pipeline_chbmit import ChbmitPreprocessingPipeline
from src.config import load_config
from src.utils.logger import print_header, print_success, print_error, console
from src.utils.io import save_numpy, ensure_dir


def add_noise(signal: np.ndarray, noise_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    向信号添加高斯白噪声
    
    参数:
        signal: 输入信号 (n_channels, n_times)
        noise_ratio: 噪声比例 (0.1 ~ 1.0)
        rng: 随机数生成器
        
    返回:
        添加噪声后的信号
    """
    signal = np.asarray(signal, dtype=np.float64)
    # 计算标准差
    scale = float(np.std(signal))
    if scale == 0.0:
        scale = 1.0  # 防止零标准差
    noise = rng.normal(loc=0.0, scale=noise_ratio * scale, size=signal.shape)
    return signal + noise


def main():
    parser = argparse.ArgumentParser(
        description="生成不同噪声比例下的样本间相似矩阵"
    )
    parser.add_argument("--config", default=str(project_root / "config/config_chbmit.yaml"),
                        help="配置文件路径")
    parser.add_argument("--data_dir", default=str(project_root),
                        help="数据根目录（包含 chb* 子目录）")
    parser.add_argument("--output_base", default=str(project_root / "output_noise"),
                        help="输出基础目录（将自动创建按噪声比例的子目录）")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="指定要处理的受试者，如 chb01 chb02，不指定则处理所有 chb*")
    parser.add_argument("--noise_ratios", nargs="+", type=float, default=None,
                        help="自定义噪声比例列表，如 0.1 0.2 ... 1.0，若不指定则使用默认 0.1~1.0 间隔0.1")
    args = parser.parse_args()

    # 确定噪声比例列表
    if args.noise_ratios is not None:
        noise_ratios = sorted(args.noise_ratios)
    else:
        noise_ratios = [round(i * 0.1, 1) for i in range(1, 11)]  # 0.1 ~ 1.0

    print_header("高斯白噪声鲁棒性测试")
    console.print(f"数据目录: {args.data_dir}")
    console.print(f"输出基础目录: {args.output_base}")
    console.print(f"噪声比例: {noise_ratios}")
    console.print(f"受试者: {args.subjects if args.subjects else '所有 chb* 目录'}")

    # 加载配置
    config = load_config(args.config)

    # 读取参数
    sfreq = getattr(config, 'preprocessing', None)
    sfreq = getattr(sfreq, 'sampling_rate', 256) if sfreq else 256

    chb = getattr(config, 'chbmit', None)
    window_duration = getattr(chb, 'window_duration', 4.0) if chb else 4.0
    n_interictal = getattr(chb, 'n_interictal_per_seizure', 1) if chb else 1
    preictal_ex = getattr(chb, 'preictal_exclude', 0.0) if chb else 0.0
    postictal_ex = getattr(chb, 'postictal_exclude', 0.0) if chb else 0.0
    interictal_gap = getattr(chb, 'interictal_min_gap', 10.0) if chb else 10.0
    seizure_step = getattr(chb, 'seizure_window_step', 1.0) if chb else 1.0

    data_path = Path(args.data_dir)
    output_base = Path(args.output_base)
    ensure_dir(output_base)

    # 确定受试者
    if args.subjects is None:
        subjects = sorted([p.name for p in data_path.glob("chb*") if p.is_dir()])
    else:
        subjects = args.subjects
    if not subjects:
        print_error("未找到任何 chb* 目录")
        sys.exit(1)
    console.print(f"处理受试者: {subjects}")

    # 初始化随机数生成器（固定种子以便复现）
    rng = np.random.default_rng(seed=42)

    # 对每个噪声比例循环处理
    for ratio in noise_ratios:
        console.print(f"\n[bold magenta]===== 处理噪声比例: {ratio:.1f} =====[/bold magenta]")
        output_dir = output_base / f"noise_{int(ratio * 100):02d}"
        ensure_dir(output_dir)

        pipeline = ChbmitPreprocessingPipeline(config)
        all_samples = []
        all_labels = []

        for subj in subjects:
            subj_dir = data_path / subj
            console.print(f"\n[bold]Processing {subj}[/bold]")
            try:
                data_list, sfreq_list, seizures_list = load_chbmit_subject(str(subj_dir), target_sfreq=sfreq)
            except Exception as e:
                console.print(f"[red]Load {subj} failed: {e}[/red]")
                continue

            for idx, (data, fs, seizures) in enumerate(zip(data_list, sfreq_list, seizures_list)):
                console.print(f"  File {idx}: {len(seizures)} seizures")
                try:
                    segs, labs = extract_segments(
                        data, fs, seizures,
                        window_duration=window_duration,
                        preictal_duration=preictal_ex,
                        postictal_duration=postictal_ex,
                        n_interictal_per_seizure=n_interictal,
                        interictal_min_gap=interictal_gap,
                        seizure_window_step=seizure_step
                    )
                except Exception as e:
                    console.print(f"[red]  Extract failed: {e}[/red]")
                    continue
                console.print(f"  Extracted {len(segs)} segments (ictal={sum(labs)}, interictal={len(labs)-sum(labs)})")
                if len(segs) == 0:
                    continue

                # 预处理
                try:
                    segs_clean, labs_clean, _ = pipeline.run(segs, labs, fs)
                except Exception as e:
                    console.print(f"[red]  Preprocess failed: {e}[/red]")
                    continue

                # 对每个干净的窗口添加噪声，然后展平
                for win in segs_clean:
                    # 添加高斯白噪声
                    noisy_win = add_noise(win, ratio, rng)
                    all_samples.append(noisy_win.flatten())
                all_labels.extend(labs_clean)
                console.print(f"[green]  Added {len(segs_clean)} samples (total so far: {len(all_samples)})[/green]")

        if not all_samples:
            print_error(f"噪声比例 {ratio:.1f} 未生成任何样本")
            continue

        X_flat = np.array(all_samples, dtype=np.float32)
        y = np.array(all_labels, dtype=int)

        console.print(f"Total samples: {X_flat.shape[0]}, feature dimension: {X_flat.shape[1]}")

        # 计算皮尔逊相似矩阵
        console.print("Computing Pearson similarity matrix ...")
        X_centered = X_flat - X_flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_normed = X_centered / norms
        S = X_normed @ X_normed.T
        S = np.clip(S, -1, 1)

        console.print(f"Similarity matrix shape: {S.shape}")

        # 保存结果
        save_numpy(S, output_dir / "similarity_matrix.npy")
        save_numpy(y, output_dir / "labels.npy")
        np.save(output_dir / "subject_ids.npy", np.array(subjects))

        print_success(f"噪声比例 {ratio:.1f} 完成! 相似矩阵: {S.shape}, 标签: {y.shape}")
        console.print(f"保存到: {output_dir}")

    console.print("\n[bold green]所有噪声比例处理完成！[/bold green]")


if __name__ == "__main__":
    main()
