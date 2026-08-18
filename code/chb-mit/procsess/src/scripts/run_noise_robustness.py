#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noise robustness experiments for per-patient analysis."""

from __future__ import annotations

import argparse
import hashlib
import copy
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.noise.artifacts import apply_noise_condition
from src.pipeline.patient_pipeline import (  # noqa: E402
    PatientResult,
    classify_patient,
    compute_patient_feature_bundle,
    load_patient_data,
    load_patient_feature_bundle,
    persist_patient_feature_bundle,
)
from src.utils.console import (  # noqa: E402
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from src.visualization.noise_robustness_plotter import plot_noise_robustness  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOISE_EXP_ROOT = "/mnt/gs21/scratch/jiangj33/EEG_EXP/Noise"
os.environ.setdefault("NOISE_EXP_ROOT", DEFAULT_NOISE_EXP_ROOT)

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None


def _resolve_config_path(config_arg: str) -> Path:
    config_path = Path(os.path.expandvars(os.path.expanduser(config_arg)))
    if config_path.exists():
        return config_path
    return PROJECT_ROOT / config_arg


def _resolve_runtime_path(path_value: str | Path, base_dir: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if path.is_absolute():
        return path
    return base_dir / path


def _normalize_runtime_paths(config: Dict[str, Any]) -> None:
    data_cfg = config.get("data", {})
    if data_cfg.get("patients_data_dir"):
        data_cfg["patients_data_dir"] = str(_resolve_runtime_path(data_cfg["patients_data_dir"]))

    cls_cfg = config.get("classification", {})
    if cls_cfg.get("pretrained_params_root"):
        cls_cfg["pretrained_params_root"] = str(_resolve_runtime_path(cls_cfg["pretrained_params_root"]))

    output_cfg = config.get("output", {})
    if output_cfg.get("output_dir"):
        output_cfg["output_dir"] = str(_resolve_runtime_path(output_cfg["output_dir"]))
    if output_cfg.get("feature_cache_dir"):
        output_cfg["feature_cache_dir"] = str(_resolve_runtime_path(output_cfg["feature_cache_dir"]))


def _scan_patients(config: Dict[str, Any]) -> tuple[Path, list[str]]:
    data_cfg = config.get("data", {})
    patients_dir = _resolve_runtime_path(data_cfg.get("patients_data_dir", ""))
    if not patients_dir.exists():
        raise FileNotFoundError(f"患者数据目录不存在: {patients_dir}")

    patients = [p.name for p in patients_dir.iterdir() if p.is_dir() and (p / "x_data.npy").exists()]
    patients.sort()

    whitelist = data_cfg.get("patients", "all")
    whitelist = _normalize_patient_selector(whitelist)
    if whitelist != "all":
        patients = [p for p in patients if p in whitelist]
    exclude = data_cfg.get("exclude_patients", [])
    patients = [p for p in patients if p not in exclude]
    return patients_dir, patients


def _get_enabled_noise_types(config: Dict[str, Any]) -> List[str]:
    return list(config.get("noise", {}).get("enabled_noise_types", ["awgn", "emg_burst", "eog_blink"]))


def _parse_noise_types_arg(raw_value: str) -> List[str]:
    items = [item.strip() for item in str(raw_value).split(",")]
    return [item for item in items if item]


def _parse_protocols_arg(raw_value: str) -> List[str]:
    items = [item.strip() for item in str(raw_value).split(",")]
    return [item for item in items if item]


def _parse_patients_arg(raw_value: str) -> List[str]:
    items = [item.strip() for item in str(raw_value).split(",")]
    return [item for item in items if item]


def _normalize_patient_selector(selector: Any) -> str | List[str]:
    if selector == "all":
        return "all"
    if isinstance(selector, str):
        return _parse_patients_arg(selector)
    if selector is None:
        return "all"
    return list(selector)


def _condition_dir_name(noise_type: str, level: str | int | float) -> str:
    if str(level).lower() == "clean":
        return "clean"
    if noise_type == "awgn":
        return f"snr_{int(float(level))}dB"
    return str(level).lower()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    return str(obj)


def _build_result_payload(
    patient_id: str,
    labels: np.ndarray,
    feature_dim: int,
    classifier_results: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    result = PatientResult(
        patient_id=patient_id,
        n_samples=int(len(labels)),
        n_class0=int(np.sum(labels == 0)),
        n_class1=int(np.sum(labels == 1)),
        feature_dim=int(feature_dim),
        classifier_results=classifier_results,
        error="",
    )
    payload = asdict(result)
    payload["metadata"] = metadata
    return payload


def _save_result_payload(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _result_payload_path(output_dir: Path) -> Path:
    return output_dir / "result_metrics.json"


def _config_fingerprint(config: Dict[str, Any]) -> str:
    relevant_config = {
        "similarity": config.get("similarity", {}),
        "graph": config.get("graph", {}),
        "lorenz": config.get("lorenz", {}),
        "perturbation": config.get("perturbation", {}),
        "classification": config.get("classification", {}),
        "noise": config.get("noise", {}),
    }
    payload = json.dumps(relevant_config, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _use_result_cache(config: Dict[str, Any]) -> bool:
    output_cfg = config.get("output", {})
    return bool(output_cfg.get("use_result_cache", output_cfg.get("use_feature_cache", True)))


def _load_cached_classifier_results(output_dir: Path, config_fingerprint: str) -> Dict[str, Any] | None:
    result_path = _result_payload_path(output_dir)
    if not result_path.exists():
        return None
    payload = _load_json(result_path)
    metadata = payload.get("metadata", {})
    if metadata.get("config_fingerprint") != config_fingerprint:
        return None
    classifier_results = payload.get("classifier_results")
    return classifier_results if isinstance(classifier_results, dict) else None


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_classifier_results(
    patient_id: str,
    noise_type: str,
    protocol: str,
    condition: str,
    classifier_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, metrics in classifier_results.items():
        if "_" not in key:
            feature_group = "ALL"
            classifier = key
        else:
            feature_group, classifier = key.rsplit("_", 1)
        row = {
            "patient_id": patient_id,
            "noise_type": noise_type,
            "protocol": protocol,
            "condition": condition,
            "feature_group": feature_group,
            "classifier": classifier,
        }
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                row[metric_name] = float(metric_value)
        rows.append(row)
    return rows


def _compute_correlation_shift(clean_p: np.ndarray, noisy_p: np.ndarray) -> Dict[str, float]:
    delta = noisy_p - clean_p
    abs_delta = np.abs(delta)
    return {
        "mean_abs_diff": float(np.mean(abs_delta)),
        "fro_norm_diff": float(np.linalg.norm(delta, ord="fro")),
        "max_abs_diff": float(np.max(abs_delta)),
    }


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-Score", "AUC"] if c in metrics_df.columns]
    agg_map = {}
    for metric_name in metric_cols:
        agg_map[f"{metric_name}_mean"] = (metric_name, "mean")
        agg_map[f"{metric_name}_std"] = (metric_name, "std")
    grouped = metrics_df.groupby(
        ["noise_type", "protocol", "condition", "feature_group", "classifier"],
        as_index=False,
    ).agg(**agg_map)
    return grouped


def _export_tables(output_root: Path, metric_rows: List[Dict[str, Any]], corr_rows: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregated_dir = output_root / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("未生成任何分类结果。")
    metrics_df.to_csv(aggregated_dir / "metrics_long.csv", index=False)

    summary_df = _summarize_metrics(metrics_df)
    summary_df.to_csv(aggregated_dir / "summary_by_condition.csv", index=False)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(aggregated_dir / "correlation_shift.csv", index=False)

    # Also export per-noise CSV sets under aggregated/<noise_type>/.
    for noise_type in sorted(metrics_df["noise_type"].dropna().unique().tolist()):
        noise_dir = aggregated_dir / str(noise_type)
        noise_dir.mkdir(parents=True, exist_ok=True)

        metrics_noise_df = metrics_df[metrics_df["noise_type"] == noise_type]
        summary_noise_df = summary_df[summary_df["noise_type"] == noise_type]
        corr_noise_df = corr_df[corr_df["noise_type"] == noise_type] if "noise_type" in corr_df.columns else pd.DataFrame()

        metrics_noise_df.to_csv(noise_dir / "metrics_long.csv", index=False)
        summary_noise_df.to_csv(noise_dir / "summary_by_condition.csv", index=False)
        corr_noise_df.to_csv(noise_dir / "correlation_shift.csv", index=False)

    return metrics_df, summary_df


def _feature_cache_root(config: Dict[str, Any], output_root: Path) -> Path:
    feature_cache_dir = config.get("output", {}).get("feature_cache_dir")
    if feature_cache_dir:
        return Path(feature_cache_dir)
    return output_root / "_feature_cache"


def _clean_cache_dir(feature_root: Path, patient_id: str) -> Path:
    return feature_root / patient_id / "clean"


def _condition_cache_dir(feature_root: Path, patient_id: str, noise_type: str, condition: str) -> Path:
    return feature_root / patient_id / noise_type / condition


def _noise_meta_path(bundle_dir: Path) -> Path:
    return bundle_dir / "noise_metadata.json"


def _has_feature_bundle_cache(bundle_dir: Path, require_similarity: bool = True) -> bool:
    required_files = ["features.npy", "labels.npy", "meta.json"]
    if require_similarity:
        required_files.append("similarity.npy")
    return all((bundle_dir / filename).exists() for filename in required_files)


def _iter_noise_conditions(
    noise_cfg: Dict[str, Any],
    noise_types: List[str],
) -> List[tuple[str, Dict[str, Any], int, Any, str]]:
    conditions: List[tuple[str, Dict[str, Any], int, Any, str]] = []
    for noise_type in noise_types:
        type_cfg = noise_cfg.get(noise_type, {})
        levels = list(type_cfg.get("levels", ["clean"]))
        for level_idx, level in enumerate(levels):
            condition_name = _condition_dir_name(noise_type, level)
            conditions.append((noise_type, type_cfg, level_idx, level, condition_name))
    return conditions


def _available_noise_condition_labels(config: Dict[str, Any], include_clean: bool = False) -> List[str]:
    noise_cfg = config.get("noise", {})
    labels: List[str] = []
    for noise_type, _, _, _, condition_name in _iter_noise_conditions(noise_cfg, _get_enabled_noise_types(config)):
        if condition_name == "clean" and not include_clean:
            continue
        labels.append(f"{noise_type}:{condition_name}")
    return labels


def _apply_noise_condition_filter(config: Dict[str, Any], selector: str) -> str:
    raw_selector = selector.strip()
    if not raw_selector:
        return ""
    if ":" not in raw_selector:
        raise ValueError("噪声档位格式应为 noise_type:level，例如 awgn:snr_30dB 或 eog_blink:s03")

    selected_noise_type, selected_level = [part.strip() for part in raw_selector.split(":", 1)]
    selected_level_norm = selected_level.lower()
    noise_cfg = config.setdefault("noise", {})
    type_cfg = noise_cfg.get(selected_noise_type)
    if not isinstance(type_cfg, dict):
        raise ValueError(f"不支持的噪声类型: {selected_noise_type}")

    for level in type_cfg.get("levels", ["clean"]):
        condition_name = _condition_dir_name(selected_noise_type, level)
        level_norm = str(level).lower()
        if selected_level_norm in {condition_name.lower(), level_norm}:
            noise_cfg["enabled_noise_types"] = [selected_noise_type]
            type_cfg["levels"] = [level]
            return f"{selected_noise_type}:{condition_name}"

    supported = ", ".join(_available_noise_condition_labels(config, include_clean=True))
    raise ValueError(f"不支持的噪声档位: {raw_selector}。可用档位: {supported}")


def _persist_bundle_if_missing(bundle, output_dir: Path) -> None:
    if not (output_dir / "features.npy").exists():
        persist_patient_feature_bundle(bundle, output_dir)


def _make_stage_progress() -> Progress:
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("calc:{task.fields[computed]} cache:{task.fields[cached]}"),
        console=console,
    )


def _noise_steps_per_patient(config: Dict[str, Any]) -> int:
    noise_cfg = config.get("noise", {})
    noise_types = _get_enabled_noise_types(config)
    return max(1, sum(len(noise_cfg.get(noise_type, {}).get("levels", ["clean"])) for noise_type in noise_types))


def _feature_steps_per_patient(config: Dict[str, Any]) -> int:
    noise_cfg = config.get("noise", {})
    noise_types = _get_enabled_noise_types(config)
    non_clean_steps = sum(
        1
        for noise_type in noise_types
        for level in noise_cfg.get(noise_type, {}).get("levels", ["clean"])
        if str(level).lower() != "clean"
    )
    return max(1, 1 + non_clean_steps)


def _resolve_parallel_plan(config: Dict[str, Any], n_patients: int) -> Dict[str, int | bool | float]:
    par_cfg = config.get("parallel", {})
    total_cpus = max(1, int(os.cpu_count() or 1))

    target = float(par_cfg.get("cpu_utilization_target", 0.8))
    target = min(max(target, 0.1), 1.0)

    reserve_cores_cfg = par_cfg.get("reserve_cores")
    if reserve_cores_cfg is None:
        reserve_cores = max(1, int(math.ceil(total_cpus * (1.0 - target))))
    else:
        reserve_cores = max(0, int(reserve_cores_cfg))

    usable_cores = max(1, total_cpus - reserve_cores)
    patient_parallel = bool(par_cfg.get("patient_parallel", True)) and n_patients > 1
    max_patient_workers_cfg = int(par_cfg.get("max_patient_workers", usable_cores))
    patient_workers = min(max(1, max_patient_workers_cfg), usable_cores, max(1, n_patients))
    if not patient_parallel:
        patient_workers = 1

    allow_nested = bool(par_cfg.get("allow_nested_parallel", False))
    feature_inner_workers_cfg = par_cfg.get("feature_inner_workers")
    if feature_inner_workers_cfg is None:
        feature_inner_workers = max(1, usable_cores // patient_workers)
    else:
        feature_inner_workers = max(1, int(feature_inner_workers_cfg))
    if patient_workers > 1 and not allow_nested:
        feature_inner_workers = 1
    feature_inner_workers = min(feature_inner_workers, usable_cores)

    blas_threads_cfg = par_cfg.get("blas_threads_per_worker")
    if blas_threads_cfg is None:
        blas_threads_per_worker = max(1, usable_cores // patient_workers)
    else:
        blas_threads_per_worker = max(1, int(blas_threads_cfg))

    return {
        "total_cpus": total_cpus,
        "reserve_cores": reserve_cores,
        "usable_cores": usable_cores,
        "patient_parallel": patient_parallel,
        "patient_workers": patient_workers,
        "feature_inner_workers": feature_inner_workers,
        "blas_threads_per_worker": blas_threads_per_worker,
        "cpu_utilization_target": target,
    }


def _make_worker_config(config: Dict[str, Any], plan: Dict[str, int | bool | float]) -> Dict[str, Any]:
    worker_config = copy.deepcopy(config)
    worker_config.setdefault("parallel", {})
    worker_config["parallel"]["feature_inner_workers"] = int(plan["feature_inner_workers"])
    return worker_config


@contextmanager
def _cpu_limited_threads(max_threads: int):
    max_threads = max(1, int(max_threads))
    env_names = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    previous = {name: os.environ.get(name) for name in env_names}
    try:
        for name in env_names:
            os.environ[name] = str(max_threads)
        if threadpool_limits is not None:
            with threadpool_limits(limits=max_threads):
                yield
        else:
            yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _print_parallel_plan(stage_name: str, plan: Dict[str, int | bool | float]) -> None:
    print_info(
        f"{stage_name} 并行计划: patient_workers={plan['patient_workers']}, "
        f"feature_inner_workers={plan['feature_inner_workers']}, "
        f"blas_threads_per_worker={plan['blas_threads_per_worker']}, "
        f"reserve_cores={plan['reserve_cores']}/{plan['total_cpus']}"
    )


def _process_feature_patient(
    patient_idx: int,
    patient_id: str,
    patient_dir: Path,
    config: Dict[str, Any],
    output_root: Path,
    plan: Dict[str, int | bool | float],
) -> Dict[str, Any]:
    noise_cfg = config.get("noise", {})
    noise_types = _get_enabled_noise_types(config)
    base_seed = int(noise_cfg.get("random_seed", config.get("experiment", {}).get("random_seed", 42)))
    use_cache = bool(config.get("output", {}).get("use_feature_cache", True))
    feature_root = _feature_cache_root(config, output_root)
    computed_conditions = 0
    cached_conditions = 0

    with _cpu_limited_threads(int(plan["blas_threads_per_worker"])):
        x_clean, y = load_patient_data(str(patient_dir))
        if int(np.sum(y == 0)) < 5 or int(np.sum(y == 1)) < 5:
            return {
                "patient_id": patient_id,
                "steps": _feature_steps_per_patient(config),
                "skipped": True,
                "warning": f"{patient_id} 样本不足，已跳过",
                "computed": computed_conditions,
                "cached": cached_conditions,
            }

        clean_dir = _clean_cache_dir(feature_root, patient_id)
        if use_cache and _has_feature_bundle_cache(clean_dir, require_similarity=True):
            cached_conditions += 1
        else:
            compute_patient_feature_bundle(
                x_clean,
                y,
                config,
                output_dir=clean_dir,
                use_cache=use_cache,
                return_similarity=True,
            )
            computed_conditions += 1
        _save_json(
            _noise_meta_path(clean_dir),
            {
                "noise_type": "clean",
                "level": "clean",
                "random_seed": base_seed + patient_idx,
                "applied": False,
            },
        )

        for noise_type, type_cfg, level_idx, level, condition_name in _iter_noise_conditions(noise_cfg, noise_types):
            if condition_name == "clean":
                continue

            condition_seed = base_seed + patient_idx * 1000 + level_idx * 100
            bundle_dir = _condition_cache_dir(feature_root, patient_id, noise_type, condition_name)
            meta_path = _noise_meta_path(bundle_dir)
            if use_cache and _has_feature_bundle_cache(bundle_dir, require_similarity=True) and meta_path.exists():
                cached_conditions += 1
                continue

            noisy_x, artifact_meta = apply_noise_condition(
                x_clean,
                noise_type=noise_type,
                level=level,
                noise_config=type_cfg,
                random_seed=condition_seed,
            )
            compute_patient_feature_bundle(
                noisy_x,
                y,
                config,
                output_dir=bundle_dir,
                use_cache=use_cache,
                return_similarity=True,
            )
            _save_json(_noise_meta_path(bundle_dir), artifact_meta)
            computed_conditions += 1

    return {
        "patient_id": patient_id,
        "steps": _feature_steps_per_patient(config),
        "skipped": False,
        "computed": computed_conditions,
        "cached": cached_conditions,
    }


def _process_classification_patient(
    patient_idx: int,
    patient_id: str,
    config: Dict[str, Any],
    output_root: Path,
    plan: Dict[str, int | bool | float],
) -> Dict[str, Any]:
    noise_cfg = config.get("noise", {})
    protocols = list(noise_cfg.get("protocols", ["clean_to_noisy", "noisy_to_noisy"]))
    noise_types = _get_enabled_noise_types(config)
    base_seed = int(noise_cfg.get("random_seed", config.get("experiment", {}).get("random_seed", 42)))
    include_clean_condition = bool(noise_cfg.get("include_clean_condition", True))
    feature_root = _feature_cache_root(config, output_root)
    result_cache_enabled = _use_result_cache(config)
    config_fingerprint = _config_fingerprint(config)
    computed_conditions = 0
    cached_conditions = 0

    metric_rows: List[Dict[str, Any]] = []
    corr_rows: List[Dict[str, Any]] = []

    with _cpu_limited_threads(int(plan["blas_threads_per_worker"])):
        clean_dir = _clean_cache_dir(feature_root, patient_id)
        if not clean_dir.exists():
            raise FileNotFoundError(f"缺少 clean 特征缓存，请先运行 features 阶段: {clean_dir}")

        clean_bundle = load_patient_feature_bundle(clean_dir, config, include_similarity=True)
        y = clean_bundle.labels
        if int(np.sum(y == 0)) < 5 or int(np.sum(y == 1)) < 5:
            return {
                "patient_id": patient_id,
                "steps": _noise_steps_per_patient(config),
                "skipped": True,
                "warning": f"{patient_id} 样本不足，已跳过",
                "metric_rows": metric_rows,
                "corr_rows": corr_rows,
                "computed": computed_conditions,
                "cached": cached_conditions,
            }

        clean_results: Dict[str, Any] | None = None

        for noise_type in noise_types:
            if include_clean_condition:
                for protocol in protocols:
                    clean_output_dir = output_root / noise_type / protocol / "per_patient" / patient_id / "clean"
                    clean_meta = {
                        "noise_type": noise_type,
                        "protocol": protocol,
                        "condition": "clean",
                        "random_seed": base_seed + patient_idx,
                        "config_fingerprint": config_fingerprint,
                        "artifact_metadata": {"noise_type": noise_type, "level": "clean", "applied": False},
                    }
                    cached_results = (
                        _load_cached_classifier_results(clean_output_dir, config_fingerprint) if result_cache_enabled else None
                    )
                    if cached_results is not None:
                        classifier_results = cached_results
                        cached_conditions += 1
                    else:
                        if clean_results is None:
                            clean_results = classify_patient(
                                clean_bundle.features,
                                y,
                                config,
                                clean_bundle.feature_dims,
                                patient_id=patient_id,
                            )
                        classifier_results = clean_results
                        _persist_bundle_if_missing(clean_bundle, clean_output_dir)
                        _save_result_payload(
                            clean_output_dir,
                            _build_result_payload(
                                patient_id=patient_id,
                                labels=y,
                                feature_dim=clean_bundle.features.shape[1],
                                classifier_results=classifier_results,
                                metadata=clean_meta,
                            ),
                        )
                        computed_conditions += 1
                    metric_rows.extend(_flatten_classifier_results(patient_id, noise_type, protocol, "clean", classifier_results))
                    _save_json(
                        clean_output_dir / "correlation_shift.json",
                        {
                            "patient_id": patient_id,
                            "noise_type": noise_type,
                            "condition": "clean",
                            "mean_abs_diff": 0.0,
                            "fro_norm_diff": 0.0,
                            "max_abs_diff": 0.0,
                        },
                    )

                corr_rows.append(
                    {
                        "patient_id": patient_id,
                        "noise_type": noise_type,
                        "condition": "clean",
                        "mean_abs_diff": 0.0,
                        "fro_norm_diff": 0.0,
                        "max_abs_diff": 0.0,
                    },
                )

            for _, _, level_idx, _, condition_name in _iter_noise_conditions(noise_cfg, [noise_type]):
                if condition_name == "clean":
                    continue

                noisy_cache_dir = _condition_cache_dir(feature_root, patient_id, noise_type, condition_name)
                if not noisy_cache_dir.exists():
                    raise FileNotFoundError(f"缺少噪声特征缓存，请先运行 features 阶段: {noisy_cache_dir}")

                noisy_bundle = load_patient_feature_bundle(noisy_cache_dir, config, include_similarity=True)
                artifact_meta = _load_json(_noise_meta_path(noisy_cache_dir))
                corr_stats = _compute_correlation_shift(clean_bundle.similarity, noisy_bundle.similarity)
                corr_rows.append(
                    {
                        "patient_id": patient_id,
                        "noise_type": noise_type,
                        "condition": condition_name,
                        **corr_stats,
                    }
                )

                condition_seed = base_seed + patient_idx * 1000 + level_idx * 100
                for protocol in protocols:
                    condition_dir = output_root / noise_type / protocol / "per_patient" / patient_id / condition_name
                    metadata = {
                        "noise_type": noise_type,
                        "protocol": protocol,
                        "condition": condition_name,
                        "random_seed": condition_seed,
                        "config_fingerprint": config_fingerprint,
                        "artifact_metadata": artifact_meta,
                    }
                    cached_results = (
                        _load_cached_classifier_results(condition_dir, config_fingerprint) if result_cache_enabled else None
                    )
                    if cached_results is not None:
                        classifier_results = cached_results
                        cached_conditions += 1
                    else:
                        _persist_bundle_if_missing(noisy_bundle, condition_dir)
                        classifier_results = _run_single_condition(
                            patient_id=patient_id,
                            labels=y,
                            clean_bundle=clean_bundle,
                            noisy_bundle=noisy_bundle,
                            protocol=protocol,
                            config=config,
                            output_dir=condition_dir,
                            metadata=metadata,
                        )
                        computed_conditions += 1
                    _save_json(
                        condition_dir / "correlation_shift.json",
                        {
                            "patient_id": patient_id,
                            "noise_type": noise_type,
                            "condition": condition_name,
                            **corr_stats,
                        },
                    )
                    metric_rows.extend(
                        _flatten_classifier_results(
                            patient_id=patient_id,
                            noise_type=noise_type,
                            protocol=protocol,
                            condition=condition_name,
                            classifier_results=classifier_results,
                        )
                    )

    return {
        "patient_id": patient_id,
        "steps": _noise_steps_per_patient(config),
        "skipped": False,
        "metric_rows": metric_rows,
        "corr_rows": corr_rows,
        "computed": computed_conditions,
        "cached": cached_conditions,
    }


def _run_feature_stage(
    config: Dict[str, Any],
    patients_dir: Path,
    patients: List[str],
    output_root: Path,
) -> None:
    feature_root = _feature_cache_root(config, output_root)
    feature_root.mkdir(parents=True, exist_ok=True)
    plan = _resolve_parallel_plan(config, len(patients))
    worker_config = _make_worker_config(config, plan)
    computed_total = 0
    cached_total = 0

    with _make_stage_progress() as progress:
        task = progress.add_task("Features", total=len(patients), computed=0, cached=0)
        if plan["patient_parallel"]:
            with ProcessPoolExecutor(max_workers=int(plan["patient_workers"])) as executor:
                futures = {
                    executor.submit(
                        _process_feature_patient,
                        patient_idx,
                        patient_id,
                        patients_dir / patient_id,
                        worker_config,
                        output_root,
                        plan,
                    ): patient_id
                    for patient_idx, patient_id in enumerate(patients)
                }
                for future in as_completed(futures):
                    res = future.result()
                    if res.get("warning"):
                        print_warning(str(res["warning"]))
                    computed_total += int(res.get("computed", 0))
                    cached_total += int(res.get("cached", 0))
                    progress.update(task, computed=computed_total, cached=cached_total)
                    progress.advance(task)
        else:
            for patient_idx, patient_id in enumerate(patients):
                res = _process_feature_patient(
                    patient_idx,
                    patient_id,
                    patients_dir / patient_id,
                    worker_config,
                    output_root,
                    plan,
                )
                if res.get("warning"):
                    print_warning(str(res["warning"]))
                computed_total += int(res.get("computed", 0))
                cached_total += int(res.get("cached", 0))
                progress.update(task, computed=computed_total, cached=cached_total)
                progress.advance(task)

    print_success(f"Features done: {feature_root}")


def _run_single_condition(
    patient_id: str,
    labels: np.ndarray,
    clean_bundle,
    noisy_bundle,
    protocol: str,
    config: Dict[str, Any],
    output_dir: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    if protocol == "clean_to_noisy":
        classifier_results = classify_patient(
            clean_bundle.features,
            labels,
            config,
            clean_bundle.feature_dims,
            output_dir=output_dir,
            eval_features=noisy_bundle.features,
            evaluation_mode="train_clean_test_noisy",
            patient_id=patient_id,
        )
    else:
        classifier_results = classify_patient(
            noisy_bundle.features,
            labels,
            config,
            noisy_bundle.feature_dims,
            output_dir=output_dir,
            patient_id=patient_id,
        )

    payload = _build_result_payload(
        patient_id=patient_id,
        labels=labels,
        feature_dim=noisy_bundle.features.shape[1],
        classifier_results=classifier_results,
        metadata=metadata,
    )
    _save_result_payload(output_dir, payload)
    return classifier_results


def _aggregate_existing_results(output_root: Path) -> None:
    metric_rows: List[Dict[str, Any]] = []
    corr_by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for result_path in sorted(output_root.glob("*/*/per_patient/*/*/result_metrics.json")):
        payload = _load_json(result_path)
        metadata = payload.get("metadata", {})
        patient_id = str(payload.get("patient_id") or metadata.get("patient_id") or result_path.parents[1].name)
        noise_type = str(metadata.get("noise_type") or result_path.parents[4].name)
        protocol = str(metadata.get("protocol") or result_path.parents[3].name)
        condition = str(metadata.get("condition") or result_path.parent.name)
        classifier_results = payload.get("classifier_results", {})
        if isinstance(classifier_results, dict):
            metric_rows.extend(_flatten_classifier_results(patient_id, noise_type, protocol, condition, classifier_results))

        corr_path = result_path.parent / "correlation_shift.json"
        if corr_path.exists():
            corr_payload = _load_json(corr_path)
        elif condition == "clean":
            corr_payload = {
                "patient_id": patient_id,
                "noise_type": noise_type,
                "condition": "clean",
                "mean_abs_diff": 0.0,
                "fro_norm_diff": 0.0,
                "max_abs_diff": 0.0,
            }
        else:
            corr_payload = {}
        if corr_payload:
            corr_by_key[(patient_id, noise_type, condition)] = corr_payload

    _, summary_df = _export_tables(output_root, metric_rows, list(corr_by_key.values()))
    plot_noise_robustness(summary_df, output_root / "figures")
    print_success(f"Aggregate done: {output_root}")


def _run_classification_stage(
    config: Dict[str, Any],
    patients: List[str],
    output_root: Path,
    aggregate: bool = True,
) -> None:
    plan = _resolve_parallel_plan(config, len(patients))
    worker_config = _make_worker_config(config, plan)
    metric_rows: List[Dict[str, Any]] = []
    corr_rows: List[Dict[str, Any]] = []
    computed_total = 0
    cached_total = 0

    with _make_stage_progress() as progress:
        task = progress.add_task("Classify", total=len(patients), computed=0, cached=0)
        if plan["patient_parallel"]:
            with ProcessPoolExecutor(max_workers=int(plan["patient_workers"])) as executor:
                futures = {
                    executor.submit(
                        _process_classification_patient,
                        patient_idx,
                        patient_id,
                        worker_config,
                        output_root,
                        plan,
                    ): patient_id
                    for patient_idx, patient_id in enumerate(patients)
                }
                for future in as_completed(futures):
                    res = future.result()
                    if res.get("warning"):
                        print_warning(str(res["warning"]))
                    metric_rows.extend(res.get("metric_rows", []))
                    corr_rows.extend(res.get("corr_rows", []))
                    computed_total += int(res.get("computed", 0))
                    cached_total += int(res.get("cached", 0))
                    progress.update(task, computed=computed_total, cached=cached_total)
                    progress.advance(task)
        else:
            for patient_idx, patient_id in enumerate(patients):
                res = _process_classification_patient(
                    patient_idx,
                    patient_id,
                    worker_config,
                    output_root,
                    plan,
                )
                if res.get("warning"):
                    print_warning(str(res["warning"]))
                metric_rows.extend(res.get("metric_rows", []))
                corr_rows.extend(res.get("corr_rows", []))
                computed_total += int(res.get("computed", 0))
                cached_total += int(res.get("cached", 0))
                progress.update(task, computed=computed_total, cached=cached_total)
                progress.advance(task)

    if aggregate:
        _, summary_df = _export_tables(output_root, metric_rows, corr_rows)
        plot_noise_robustness(summary_df, output_root / "figures")
    else:
        print_info("已跳过本任务内全局汇总；请在所有 Slurm array task 完成后运行 --stage aggregate。")
    print_success(f"Classify done: {output_root}")


def run_noise_robustness_pipeline(config: Dict[str, Any], stage: str = "all", aggregate: bool = True) -> None:
    print_header("Noise Robustness Analysis", subtitle="AWGN / EMG-like burst / EOG-like blink")
    output_root = _resolve_runtime_path(config.get("output", {}).get("output_dir", "outputs/Noise_Analysis"))
    output_root.mkdir(parents=True, exist_ok=True)

    stage = stage.lower()
    if stage == "aggregate":
        _aggregate_existing_results(output_root)
        return

    patients_dir, patients = _scan_patients(config)
    print_info(f"待处理患者数: {len(patients)}")

    if stage in {"features", "all"}:
        _run_feature_stage(config, patients_dir, patients, output_root)
    if stage in {"classify", "all"}:
        _run_classification_stage(config, patients, output_root, aggregate=aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise robustness analysis")
    parser.add_argument("--config", type=str, default="configs/noise_robustness.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="", help="覆盖配置中的输出目录")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "features", "classify", "aggregate"],
        help="运行阶段：all/features/classify/aggregate",
    )
    parser.add_argument(
        "--noise-types",
        type=str,
        default="",
        help="仅运行指定噪声类型，逗号分隔，例如: awgn 或 awgn,emg_burst",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default="",
        help="仅运行指定患者，逗号分隔，例如: chb01 或 chb01,chb02",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default="",
        help="仅运行指定评估协议，逗号分隔，可选: clean_to_noisy,noisy_to_noisy",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="",
        help="覆盖配置中的特征缓存目录",
    )
    parser.add_argument(
        "--noise-condition",
        type=str,
        default="",
        help="仅运行一个噪声档位，格式 noise_type:level，例如 awgn:snr_30dB 或 emg_burst:s03",
    )
    parser.add_argument(
        "--list-noise-conditions",
        action="store_true",
        help="列出当前配置展开后的非 clean 噪声档位后退出",
    )
    parser.add_argument(
        "--list-patients",
        action="store_true",
        help="列出当前配置可用患者后退出",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="分类阶段不写全局汇总，适用于 Slurm array task；全部完成后运行 --stage aggregate",
    )
    parser.add_argument(
        "--skip-clean-condition",
        action="store_true",
        help="分类阶段不写 clean 基线结果，适用于已经预先生成 clean 结果的 Slurm array task",
    )
    parser.add_argument(
        "--feature-inner-workers",
        type=int,
        default=0,
        help="覆盖并行配置 parallel.feature_inner_workers（常用于 Slurm 按核数设置）",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    if not config_path.exists():
        print_error(f"配置文件未找到: {args.config}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config.setdefault("output", {})
        config["output"]["output_dir"] = args.output_dir
        print_info(f"本次输出目录: {args.output_dir}")

    if args.feature_cache_dir:
        config.setdefault("output", {})
        config["output"]["feature_cache_dir"] = args.feature_cache_dir
        print_info(f"本次特征缓存目录: {args.feature_cache_dir}")

    if args.feature_inner_workers and args.feature_inner_workers > 0:
        config.setdefault("parallel", {})
        config["parallel"]["feature_inner_workers"] = int(args.feature_inner_workers)
        print_info(f"本次 feature_inner_workers: {args.feature_inner_workers}")

    if args.noise_types:
        selected_noise_types = _parse_noise_types_arg(args.noise_types)
        supported_noise_types = set(config.get("noise", {}).keys()) | {"awgn", "emg_burst", "eog_blink"}
        invalid_noise_types = [item for item in selected_noise_types if item not in supported_noise_types]
        if invalid_noise_types:
            print_error(f"不支持的噪声类型: {', '.join(invalid_noise_types)}")
            return
        config.setdefault("noise", {})
        config["noise"]["enabled_noise_types"] = selected_noise_types
        print_info(f"本次仅运行噪声类型: {', '.join(selected_noise_types)}")

    if args.patients:
        selected_patients = _parse_patients_arg(args.patients)
        config.setdefault("data", {})
        config["data"]["patients"] = selected_patients
        print_info(f"本次仅运行患者: {', '.join(selected_patients)}")

    if args.protocols:
        selected_protocols = _parse_protocols_arg(args.protocols)
        supported_protocols = {"clean_to_noisy", "noisy_to_noisy"}
        invalid_protocols = [item for item in selected_protocols if item not in supported_protocols]
        if invalid_protocols:
            print_error(f"不支持的评估协议: {', '.join(invalid_protocols)}")
            return
        config.setdefault("noise", {})
        config["noise"]["protocols"] = selected_protocols
        print_info(f"本次仅运行评估协议: {', '.join(selected_protocols)}")

    _normalize_runtime_paths(config)

    if args.list_noise_conditions:
        for label in _available_noise_condition_labels(config):
            print(label)
        return

    if args.list_patients:
        _, patients = _scan_patients(config)
        for patient_id in patients:
            print(patient_id)
        return

    if args.noise_condition:
        try:
            selected_condition = _apply_noise_condition_filter(config, args.noise_condition)
        except ValueError as exc:
            print_error(str(exc))
            return
        print_info(f"本次仅运行噪声档位: {selected_condition}")

    if args.skip_clean_condition:
        config.setdefault("noise", {})
        config["noise"]["include_clean_condition"] = False

    run_noise_robustness_pipeline(config, stage=args.stage, aggregate=not args.skip_aggregate)


if __name__ == "__main__":
    main()
