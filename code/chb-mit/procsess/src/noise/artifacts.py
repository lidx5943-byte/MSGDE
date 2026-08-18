"""Noise generators for robustness analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


EPS = 1e-8
EMG_LEVEL_SCALES = {"mild": 0.25, "moderate": 0.5, "severe": 1.0}
EOG_LEVEL_SCALES = {"mild": 0.5, "moderate": 1.0, "severe": 2.0}


def _make_rng(random_seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(random_seed)


def _ensure_2d(x_data: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_data, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D input (N, T), got shape {x_arr.shape}")
    return x_arr


def infer_channel_indices(num_samples: int, n_channels: int = 18) -> np.ndarray:
    """Infer per-row channel indices from flattened patient data."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    return (np.arange(num_samples, dtype=np.int32) % n_channels).astype(np.int16)


def infer_window_indices(num_samples: int, n_channels: int = 18) -> np.ndarray:
    """Infer window indices assuming rows are ordered channel-by-channel."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    return (np.arange(num_samples, dtype=np.int32) // n_channels).astype(np.int32)


def add_awgn_by_snr(
    x_data: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Inject independent AWGN to each row with target SNR."""
    x_arr = _ensure_2d(x_data)
    if rng is None:
        rng = _make_rng()

    signal_power = np.mean(np.square(x_arr), axis=1, keepdims=True)
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_std = np.sqrt(np.maximum(noise_power, EPS))
    noise = rng.normal(loc=0.0, scale=1.0, size=x_arr.shape).astype(np.float32)
    return (x_arr + noise_std * noise).astype(np.float32)


def _bandpass_noise(
    length: int,
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if length <= 3:
        return np.zeros((length,), dtype=np.float32)

    nyquist = 0.5 * sampling_rate
    high = min(highcut / nyquist, 0.999)
    low = max(lowcut / nyquist, 1e-4)
    if low >= high:
        low = max(high * 0.5, 1e-4)

    sos = butter(4, [low, high], btype="bandpass", output="sos")
    white = rng.normal(0.0, 1.0, length).astype(np.float32)
    try:
        band_limited = sosfiltfilt(sos, white).astype(np.float32)
    except ValueError:
        # Very short bursts can be shorter than sosfiltfilt's pad length.
        # Fall back to one-pass filtering so EMG-like noise generation stays robust.
        band_limited = sosfilt(sos, white).astype(np.float32)
    band_limited *= np.hanning(length).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(band_limited)) + EPS)
    return band_limited / rms


def _sample_duration(
    rng: np.random.Generator,
    sampling_rate: float,
    min_duration_sec: float,
    max_duration_sec: float,
) -> int:
    duration = rng.uniform(min_duration_sec, max_duration_sec)
    return max(1, int(round(duration * sampling_rate)))


def _build_channel_group_mask(
    channel_indices: np.ndarray,
    rng: np.random.Generator,
    min_group_channels: int,
    max_group_channels: int,
) -> np.ndarray:
    active_mask = np.zeros((len(channel_indices),), dtype=bool)
    if len(channel_indices) == 0:
        return active_mask

    unique_channels = np.unique(channel_indices)
    group_size = int(rng.integers(min_group_channels, max_group_channels + 1))
    group_size = min(group_size, len(unique_channels))
    center_channel = int(rng.choice(unique_channels))
    left_span = group_size // 2
    right_span = group_size - left_span - 1
    selected = {
        ch for ch in unique_channels
        if center_channel - left_span <= int(ch) <= center_channel + right_span
    }
    if not selected:
        selected = {center_channel}
    for idx, ch in enumerate(channel_indices):
        if int(ch) in selected:
            active_mask[idx] = True
    return active_mask


def add_emg_burst(
    x_data: np.ndarray,
    level: str,
    sampling_rate: float = 256.0,
    n_channels: int = 18,
    channel_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    min_bursts: int = 1,
    max_bursts: int = 3,
    min_duration_sec: float = 0.1,
    max_duration_sec: float = 0.5,
    lowcut: float = 20.0,
    highcut: float = 70.0,
    min_group_channels: int = 2,
    max_group_channels: int = 4,
    level_scales: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Inject localized EMG-like burst noise with shared timing."""
    x_arr = _ensure_2d(x_data)
    if rng is None:
        rng = _make_rng()
    level_scales = level_scales or EMG_LEVEL_SCALES
    if level not in level_scales:
        raise ValueError(f"Unsupported EMG level: {level}")

    num_samples, num_timepoints = x_arr.shape
    if channel_indices is None:
        channel_indices = infer_channel_indices(num_samples, n_channels)
    channel_indices = np.asarray(channel_indices, dtype=np.int32)
    window_indices = infer_window_indices(num_samples, n_channels)
    noisy = x_arr.copy()
    scale = float(level_scales[level])

    for window_idx in np.unique(window_indices):
        row_mask = window_indices == window_idx
        row_indices = np.where(row_mask)[0]
        window_rows = noisy[row_indices]
        window_channels = channel_indices[row_indices]
        n_bursts = int(rng.integers(min_bursts, max_bursts + 1))
        row_rms = np.sqrt(np.mean(np.square(window_rows), axis=1) + EPS)

        for _ in range(n_bursts):
            burst_len = _sample_duration(rng, sampling_rate, min_duration_sec, max_duration_sec)
            burst_len = min(burst_len, num_timepoints)
            start = int(rng.integers(0, max(1, num_timepoints - burst_len + 1)))
            stop = start + burst_len
            burst = _bandpass_noise(burst_len, sampling_rate, lowcut, highcut, rng)
            active_channels = _build_channel_group_mask(
                window_channels,
                rng,
                min_group_channels=min_group_channels,
                max_group_channels=max_group_channels,
            )
            if not np.any(active_channels):
                active_channels[:] = True
            for local_idx, is_active in enumerate(active_channels):
                if not is_active:
                    continue
                amplitude = scale * row_rms[local_idx]
                noisy[row_indices[local_idx], start:stop] += amplitude * burst

    return noisy.astype(np.float32)


def _default_eog_weights(n_channels: int) -> np.ndarray:
    """Simplified front > mid > back channel weights."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    thirds = np.array_split(np.arange(n_channels), 3)
    weights = np.zeros((n_channels,), dtype=np.float32)
    if len(thirds) > 0:
        weights[thirds[0]] = 1.0
    if len(thirds) > 1:
        weights[thirds[1]] = 0.55
    if len(thirds) > 2:
        weights[thirds[2]] = 0.25
    return weights


def add_eog_blink(
    x_data: np.ndarray,
    level: str,
    sampling_rate: float = 256.0,
    n_channels: int = 18,
    channel_indices: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    min_blinks: int = 0,
    max_blinks: int = 2,
    min_duration_sec: float = 0.2,
    max_duration_sec: float = 0.4,
    level_scales: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Inject EOG-like blink artifacts with spatial weighting."""
    x_arr = _ensure_2d(x_data)
    if rng is None:
        rng = _make_rng()
    level_scales = level_scales or EOG_LEVEL_SCALES
    if level not in level_scales:
        raise ValueError(f"Unsupported EOG level: {level}")

    num_samples, num_timepoints = x_arr.shape
    if channel_indices is None:
        channel_indices = infer_channel_indices(num_samples, n_channels)
    channel_indices = np.asarray(channel_indices, dtype=np.int32)
    window_indices = infer_window_indices(num_samples, n_channels)
    channel_weights = _default_eog_weights(n_channels)
    noisy = x_arr.copy()
    scale = float(level_scales[level])
    time_axis = np.arange(num_timepoints, dtype=np.float32)

    for window_idx in np.unique(window_indices):
        row_mask = window_indices == window_idx
        row_indices = np.where(row_mask)[0]
        window_rows = noisy[row_indices]
        window_channels = channel_indices[row_indices]
        row_rms = np.sqrt(np.mean(np.square(window_rows), axis=1) + EPS)
        n_blinks = int(rng.integers(min_blinks, max_blinks + 1))

        for _ in range(n_blinks):
            duration = rng.uniform(min_duration_sec, max_duration_sec)
            sigma_samples = max(1.0, duration * sampling_rate / 6.0)
            center = rng.uniform(0, max(1, num_timepoints - 1))
            pulse = np.exp(-0.5 * ((time_axis - center) / sigma_samples) ** 2).astype(np.float32)
            pulse /= np.sqrt(np.mean(np.square(pulse)) + EPS)
            for local_idx, channel_id in enumerate(window_channels):
                weight = channel_weights[int(channel_id) % n_channels]
                amplitude = scale * weight * row_rms[local_idx]
                noisy[row_indices[local_idx]] += amplitude * pulse

    return noisy.astype(np.float32)


def apply_noise_condition(
    x_data: np.ndarray,
    noise_type: str,
    level: str | int | float,
    noise_config: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    channel_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the requested noise condition to z-scored time series."""
    x_arr = _ensure_2d(x_data)
    cfg = noise_config or {}
    rng = _make_rng(random_seed)
    n_channels = int(cfg.get("n_channels", 18))
    sampling_rate = float(cfg.get("sampling_rate", 256.0))

    if str(level).lower() == "clean":
        return x_arr.copy(), {
            "noise_type": str(noise_type),
            "level": "clean",
            "random_seed": random_seed,
            "applied": False,
        }

    noise_type = str(noise_type).lower()
    if noise_type == "awgn":
        noisy = add_awgn_by_snr(x_arr, float(level), rng=rng)
    elif noise_type in {"emg", "emg_burst", "emg-like"}:
        noisy = add_emg_burst(
            x_arr,
            level=str(level).lower(),
            sampling_rate=sampling_rate,
            n_channels=n_channels,
            channel_indices=channel_indices,
            rng=rng,
            min_bursts=int(cfg.get("min_bursts", 1)),
            max_bursts=int(cfg.get("max_bursts", 3)),
            min_duration_sec=float(cfg.get("min_duration_sec", 0.1)),
            max_duration_sec=float(cfg.get("max_duration_sec", 0.5)),
            lowcut=float(cfg.get("lowcut", 20.0)),
            highcut=float(cfg.get("highcut", 70.0)),
            min_group_channels=int(cfg.get("min_group_channels", 2)),
            max_group_channels=int(cfg.get("max_group_channels", 4)),
            level_scales=cfg.get("amplitude_scales"),
        )
        noise_type = "emg_burst"
    elif noise_type in {"eog", "eog_blink", "blink"}:
        noisy = add_eog_blink(
            x_arr,
            level=str(level).lower(),
            sampling_rate=sampling_rate,
            n_channels=n_channels,
            channel_indices=channel_indices,
            rng=rng,
            min_blinks=int(cfg.get("min_blinks", 0)),
            max_blinks=int(cfg.get("max_blinks", 2)),
            min_duration_sec=float(cfg.get("min_duration_sec", 0.2)),
            max_duration_sec=float(cfg.get("max_duration_sec", 0.4)),
            level_scales=cfg.get("amplitude_scales"),
        )
        noise_type = "eog_blink"
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    metadata = {
        "noise_type": noise_type,
        "level": level,
        "random_seed": random_seed,
        "applied": True,
        "sampling_rate": sampling_rate,
        "n_channels": n_channels,
    }
    return noisy.astype(np.float32), metadata
