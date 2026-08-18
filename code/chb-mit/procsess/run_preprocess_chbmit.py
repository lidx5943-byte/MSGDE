#!/usr/bin/env python3
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root/"config/config_chbmit.yaml"))
    parser.add_argument("--data_dir", default=str(project_root))
    parser.add_argument("--output_dir", default=str(project_root/"output"))
    parser.add_argument("--subjects", nargs="+", default=None)
    args = parser.parse_args()

    print_header("CHB-MIT: Sample-level Similarity Matrix (Pearson)")
    console.print(f"Data: {args.data_dir}\nOutput: {args.output_dir}")

    config = load_config(args.config)

    # 读取参数，若配置缺失则使用默认值
    sfreq = getattr(config, 'preprocessing', None)
    sfreq = getattr(sfreq, 'sampling_rate', 256) if sfreq else 256

    chb = getattr(config, 'chbmit', None)
    window_duration = getattr(chb, 'window_duration', 4.0) if chb else 4.0
    n_interictal = getattr(chb, 'n_interictal_per_seizure', 1) if chb else 1
    preictal_ex = getattr(chb, 'preictal_exclude', 0.0) if chb else 0.0
    postictal_ex = getattr(chb, 'postictal_exclude', 0.0) if chb else 0.0
    interictal_gap = getattr(chb, 'interictal_min_gap', 10.0) if chb else 10.0
    # 新增：发作窗口滑动步长（秒），默认1.0，若设为0则回退到只取中心点
    seizure_step = getattr(chb, 'seizure_window_step', 1.0) if chb else 1.0

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir)
    ensure_dir(out_path)

    if args.subjects is None:
        subjects = sorted([p.name for p in data_path.glob("chb*") if p.is_dir()])
    else:
        subjects = args.subjects
    if not subjects:
        print_error("No chb* directories found")
        sys.exit(1)
    console.print(f"Subjects: {subjects}")

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
                    seizure_window_step=seizure_step   # 传递新参数
                )
            except Exception as e:
                console.print(f"[red]  Extract failed: {e}[/red]")
                continue
            console.print(f"  Extracted {len(segs)} segments (ictal={sum(labs)}, interictal={len(labs)-sum(labs)})")
            if len(segs) == 0:
                continue

            try:
                segs_clean, labs_clean, _ = pipeline.run(segs, labs, fs)
            except Exception as e:
                console.print(f"[red]  Preprocess failed: {e}[/red]")
                continue

            for win in segs_clean:
                all_samples.append(win.flatten())
            all_labels.extend(labs_clean)
            console.print(f"[green]  Added {len(segs_clean)} samples (total so far: {len(all_samples)})[/green]")

    if not all_samples:
        print_error("No samples generated from any subject.")
        sys.exit(1)

    X_flat = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=int)

    console.print(f"Total samples: {X_flat.shape[0]}, feature dimension: {X_flat.shape[1]}")

    console.print("Computing Pearson similarity matrix (may take a while for large N)...")
    X_centered = X_flat - X_flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X_centered / norms
    S = X_normed @ X_normed.T
    S = np.clip(S, -1, 1)

    console.print(f"Similarity matrix shape: {S.shape}")

    save_numpy(S, out_path / "similarity_matrix.npy")
    save_numpy(y, out_path / "labels.npy")
    np.save(out_path / "subject_ids.npy", np.array(subjects))

    print_success(f"Done! Similarity matrix shape: {S.shape}, labels: {y.shape}")
    console.print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
