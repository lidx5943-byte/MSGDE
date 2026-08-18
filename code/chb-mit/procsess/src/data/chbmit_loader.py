import numpy as np
import mne
from pathlib import Path
from typing import List, Tuple, Dict
from .summary_parser import SummaryParser  # 导入专用解析器

def load_chbmit_subject(subject_dir: str, target_sfreq: float = 256.0):
    subject_path = Path(subject_dir)
    if not subject_path.exists():
        raise FileNotFoundError(f'Directory not found: {subject_dir}')
    edf_files = sorted(subject_path.glob('*.edf'))
    if not edf_files:
        raise FileNotFoundError(f'No EDF files in {subject_dir}')

    # 使用 SummaryParser 解析 summary 文件
    parser = SummaryParser(str(subject_path.parent))  # 传入数据集根目录
    try:
        patient_info = parser.parse(subject_path.name)  # 例如 "chb02"
    except Exception as e:
        print(f"  [ERROR] Failed to parse summary: {e}")
        raise RuntimeError(f"No valid summary for {subject_path.name}")

    # 构建文件名到发作列表的映射
    seizure_dict = {}
    for file_info in patient_info.files:
        if file_info.n_seizures > 0:
            seizures = [(s.start_sec, s.end_sec) for s in file_info.seizures]
            seizure_dict[file_info.file_name] = seizures

    print(f"  Summary parsed: {len(seizure_dict)} files with seizures")

    data_list, sfreq_list, seizures_list = [], [], []
    for edf_path in edf_files:
        seizures = seizure_dict.get(edf_path.name, [])
        if not seizures:
            print(f'  [SKIP] {edf_path.name}: no seizures in summary')
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            if target_sfreq is not None and raw.info['sfreq'] != target_sfreq:
                raw.resample(target_sfreq, npad='auto')
            sfreq = raw.info['sfreq']
            data = raw.get_data()
            # 统一为23通道（截取或填充）
            if data.shape[0] != 23:
                if data.shape[0] > 23:
                    print(f"  [INFO] {edf_path.name}: trimming from {data.shape[0]} to 23 channels")
                    data = data[:23, :]
                else:
                    print(f"  [INFO] {edf_path.name}: padding from {data.shape[0]} to 23 channels")
                    pad = np.zeros((23 - data.shape[0], data.shape[1]), dtype=data.dtype)
                    data = np.vstack([data, pad])
            # 统一为23通道（有些文件有24通道，取前23个）
            if data.shape[0] != 23:
                print(f"  [INFO] {edf_path.name}: trimming from {data.shape[0]} to 23 channels")
                data = data[:23, :]
            n_ch = data.shape[0]
            print(f'  [OK] {edf_path.name}: {len(seizures)} seizures, channels={n_ch}')
            data_list.append(data)
            sfreq_list.append(sfreq)
            seizures_list.append(seizures)
        except Exception as e:
            print(f'  [ERROR] {edf_path.name}: {e}')
            continue
    if not data_list:
        raise RuntimeError(f'No valid EDF-annotation pairs in {subject_dir}')
    return data_list, sfreq_list, seizures_list

def extract_segments(data, sfreq, seizures, window_duration=4.0,
                     preictal_duration=0.0, postictal_duration=0.0,
                     n_interictal_per_seizure=2, interictal_min_gap=60.0,
                     seizure_window_step=1.0,
                     random_seed=42):
    np.random.seed(random_seed)
    n_channels, n_times = data.shape
    window_samples = int(window_duration * sfreq)
    step_samples = int(seizure_window_step * sfreq) if seizure_window_step and seizure_window_step > 0 else 0
    segments, labels = [], []
    seizure_mask = np.zeros(n_times, dtype=bool)
    for start, end in seizures:
        s = max(0, int((start - preictal_duration) * sfreq))
        e = min(n_times, int((end + postictal_duration) * sfreq))
        seizure_mask[s:e] = True

    possible_starts = np.arange(0, n_times - window_samples + 1, step=1)

    seizure_starts = []
    for start, end in seizures:
        if step_samples > 0:
            s_start = int(start * sfreq)
            s_end = int(end * sfreq) - window_samples
            s_start = max(0, s_start)
            s_end = min(s_end, n_times - window_samples)
            for s in range(s_start, s_end + 1, step_samples):
                if np.mean(seizure_mask[s:s+window_samples]) > 0.5:
                    seizure_starts.append(s)
        else:
            center = (start + end) / 2
            center_sample = int(center * sfreq)
            s = center_sample - window_samples // 2
            if s >= 0 and s + window_samples <= n_times:
                if np.mean(seizure_mask[s:s+window_samples]) > 0.5:
                    seizure_starts.append(s)

    if not seizure_starts:
        for start, end in seizures:
            center_sample = int(((start + end) / 2) * sfreq)
            s = center_sample - window_samples // 2
            if s >= 0 and s + window_samples <= n_times:
                if np.mean(seizure_mask[s:s+window_samples]) > 0.5:
                    seizure_starts.append(s)
                    break

    for s in seizure_starts:
        segments.append(data[:, s:s+window_samples])
        labels.append(1)

    forbidden = np.zeros(n_times, dtype=bool)
    for start, end in seizures:
        s = max(0, int((start - interictal_min_gap) * sfreq))
        e = min(n_times, int((end + interictal_min_gap) * sfreq))
        forbidden[s:e] = True

    available_starts = [s for s in possible_starts
                        if s + window_samples <= n_times and not np.any(forbidden[s:s+window_samples])]
    if not available_starts:
        available_starts = possible_starts

    n_interictal = len(seizure_starts) * n_interictal_per_seizure
    n_interictal = min(n_interictal, len(available_starts))
    if n_interictal > 0:
        chosen = np.random.choice(available_starts, size=n_interictal, replace=False)
        for s in chosen:
            segments.append(data[:, s:s+window_samples])
            labels.append(0)

    return np.array(segments), np.array(labels)
