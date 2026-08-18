# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""共识曲线：Bootstrap 基分类器池，随集成规模 k 变化评估指标。"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

from ..utils.console import console, print_header, print_step, print_success, print_error, print_info
from ..visualization.nature_style import set_nature_style, NATURE_COLORS

def split_features(X_all: np.ndarray, n_scales=10) -> Dict[str, np.ndarray]:
    """将 Perturbation 特征 (N, 10×12) 拆为几何 G、拓扑 T、动力学 D 与 Fusion。"""
    feature_sets = {}
    feature_sets["Fusion (All)"] = X_all

    n_samples = X_all.shape[0]
    dim_per_scale = 12
    dim_s = 3
    dim_t = 3
    dim_d = 6

    X_reshaped = X_all.reshape(n_samples, n_scales, dim_per_scale)

    X_s = X_reshaped[:, :, 0:dim_s].reshape(n_samples, -1)
    feature_sets["Geometric"] = X_s

    X_t = X_reshaped[:, :, dim_s:dim_s+dim_t].reshape(n_samples, -1)
    feature_sets["Topological"] = X_t

    X_d = X_reshaped[:, :, dim_s+dim_t:].reshape(n_samples, -1)
    feature_sets["Dynamics"] = X_d

    return feature_sets

def train_model_pool(
    X: np.ndarray,
    y: np.ndarray,
    pool_size: int,
    model_type: str,
    model_params: Dict[str, Any],
    seed: int
) -> List[Any]:
    """自助法重采样训练基分类器列表。"""
    models = []
    n_samples = X.shape[0]
    rng = np.random.RandomState(seed)

    for i in range(pool_size):
        indices = rng.randint(0, n_samples, n_samples)
        X_boot = X[indices]
        y_boot = y[indices]

        current_params = model_params.copy()

        if model_type == 'RF':
            clf = RandomForestClassifier(random_state=seed+i, **current_params)
        elif model_type == 'SVM':
            svm_params = {k: v for k, v in current_params.items() if k in ['C', 'kernel', 'gamma', 'degree', 'coef0']}
            clf = SVC(probability=True, random_state=seed+i, **svm_params)
        elif model_type == 'KNN':
            knn_params = {k: v for k, v in current_params.items() if k in ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric']}
            clf = KNeighborsClassifier(**knn_params)
        elif model_type == 'LR':
            lr_params = {k: v for k, v in current_params.items() if k in ['C', 'penalty', 'solver', 'l1_ratio']}
            clf = LogisticRegression(random_state=seed+i, max_iter=1000, **lr_params)
        elif model_type == 'GBDT':
            gbdt_params = {k: v for k, v in current_params.items() if k in ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'min_samples_split']}
            clf = GradientBoostingClassifier(random_state=seed+i, **gbdt_params)
        else:
            clf = RandomForestClassifier(random_state=seed+i, **current_params)

        clf.fit(X_boot, y_boot)
        models.append(clf)

    return models

def evaluate_mixed_consensus(
    pool_items: List[Tuple],
    X_test_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    max_size: int,
    n_repeats: int,
    rng: np.random.RandomState
) -> List[Dict[str, float]]:
    """混合特征池：对 (模型, 特征名) 列表做规模 k 的子集平均概率与指标。"""
    probs_all = []
    for clf, fname in pool_items:
        X_target = X_test_dict[fname]
        try:
            p = clf.predict_proba(X_target)[:, 1]
        except Exception:
            p = clf.predict(X_target)
        probs_all.append(p)
    probs_all = np.array(probs_all)

    results = []
    pool_indices = np.arange(len(pool_items))

    for k in range(1, max_size + 1):
        metrics_accum = {
            'Accuracy': [], 'AUC': [], 'F1-Score': [], 'Sensitivity': [], 'Specificity': []
        }

        current_repeats = 1 if k == len(pool_items) else n_repeats

        for _ in range(current_repeats):
            selected_idx = rng.choice(pool_indices, size=k, replace=False)
            selected_probs = probs_all[selected_idx]

            ens_prob = np.mean(selected_probs, axis=0)
            ens_pred = (ens_prob >= 0.5).astype(int)

            acc = accuracy_score(y_test, ens_pred)
            try:
                auc = roc_auc_score(y_test, ens_prob)
            except Exception:
                auc = 0.5
            f1 = f1_score(y_test, ens_pred, zero_division=0)
            sens = recall_score(y_test, ens_pred, zero_division=0)
            spec = recall_score(y_test, ens_pred, pos_label=0, zero_division=0)

            metrics_accum['Accuracy'].append(acc)
            metrics_accum['AUC'].append(auc)
            metrics_accum['F1-Score'].append(f1)
            metrics_accum['Sensitivity'].append(sens)
            metrics_accum['Specificity'].append(spec)

        for m_name, vals in metrics_accum.items():
            results.append({
                'Consensus Size': k,
                'Metric': m_name,
                'Value': np.mean(vals),
                'Std': np.std(vals)
            })

    return results

def evaluate_consensus(
    pool: List[Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_size: int,
    n_repeats: int,
    rng: np.random.RandomState
) -> List[Dict[str, float]]:
    """单特征矩阵上的共识评估（二分类概率平均）。"""
    probs_all = []
    for clf in pool:
        try:
            p = clf.predict_proba(X_test)[:, 1]
        except Exception:
            p = clf.predict(X_test)
        probs_all.append(p)
    probs_all = np.array(probs_all)

    results = []

    pool_indices = np.arange(len(pool))

    for k in range(1, max_size + 1):
        metrics_accum = {
            'Accuracy': [], 'AUC': [], 'F1-Score': [], 'Sensitivity': [], 'Specificity': []
        }

        current_repeats = 1 if k == len(pool) else n_repeats

        for _ in range(current_repeats):
            selected_idx = rng.choice(pool_indices, size=k, replace=False)
            selected_probs = probs_all[selected_idx]

            ens_prob = np.mean(selected_probs, axis=0)
            ens_pred = (ens_prob >= 0.5).astype(int)

            acc = accuracy_score(y_test, ens_pred)
            try:
                auc = roc_auc_score(y_test, ens_prob)
            except Exception:
                auc = 0.5
            f1 = f1_score(y_test, ens_pred, zero_division=0)
            sens = recall_score(y_test, ens_pred, zero_division=0)
            spec = recall_score(y_test, ens_pred, pos_label=0, zero_division=0)

            metrics_accum['Accuracy'].append(acc)
            metrics_accum['AUC'].append(auc)
            metrics_accum['F1-Score'].append(f1)
            metrics_accum['Sensitivity'].append(sens)
            metrics_accum['Specificity'].append(spec)

        for m_name, vals in metrics_accum.items():
            results.append({
                'Consensus Size': k,
                'Metric': m_name,
                'Value': np.mean(vals),
                'Std': np.std(vals)
            })

    return results

def run_consensus_analysis_pipeline(config: Dict[str, Any]):
    print_header("共识性分析")

    data_cfg = config.get("data", {})
    exp_cfg = config.get("experiment", {})
    clf_cfg = config.get("classification", {})
    out_cfg = config.get("output", {})

    feat_path = Path(data_cfg.get("perturbation_features_path"))
    labels_path = Path(data_cfg.get("labels_path"))
    output_dir = Path(out_cfg.get("save_dir", "./output/consensus"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print_step(1, 4, "加载与特征拆分")
    if not feat_path.exists() or not labels_path.exists():
        print_error("数据文件未找到")
        return

    X_all = np.load(feat_path)
    y = np.load(labels_path)
    X_all = np.nan_to_num(X_all)

    feature_sets = split_features(X_all)
    print_info(f"特征集: {list(feature_sets.keys())}")

    print_step(2, 4, "交叉验证与共识评估")

    pool_size = exp_cfg.get("pool_size", 20)
    mixed_per_class = 10
    total_mixed_size = mixed_per_class * 3

    max_size = max(exp_cfg.get("max_consensus_size", 20), total_mixed_size)

    n_repeats = exp_cfg.get("n_repeats", 10)
    n_folds = clf_cfg.get("n_folds", 5)
    model_type = clf_cfg.get("model", "RF")
    model_params = clf_cfg.get("params", {})
    seed = exp_cfg.get("random_seed", 42)

    print_info(f"基分类器类型: {model_type}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_results_df = []

    fold = 0
    for train_idx, test_idx in skf.split(X_all, y):
        fold += 1
        console.print(f"[cyan]折 {fold}/{n_folds}[/cyan]")

        y_train, y_test = y[train_idx], y[test_idx]

        fold_pools = {}
        fold_X_tests = {}

        for fs_name, X_feat in feature_sets.items():
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]
            fold_X_tests[fs_name] = X_test

            pool = train_model_pool(X_train, y_train, pool_size, model_type, model_params, seed + fold)
            fold_pools[fs_name] = pool

        name_map = {
            'S': 'Geometric',
            'T': 'Topological',
            'D': 'Dynamics',
            'Fusion': 'Fusion (All)'
        }

        experiments = [
            ('S', ['S'], 20),
            ('T', ['T'], 20),
            ('D', ['D'], 20),
            ('S+T (Mixed)', ['S', 'T'], 10),
            ('S+D (Mixed)', ['S', 'D'], 10),
            ('T+D (Mixed)', ['T', 'D'], 10),
            ('S+T+D (Mixed)', ['S', 'T', 'D'], 10)
        ]

        if 'Fusion (All)' in fold_pools:
            experiments.insert(0, ('Fusion (Feat)', ['Fusion'], 20))

        for exp_name, components, limit_per_comp in experiments:
            if not all(name_map[c] in fold_pools for c in components):
                continue

            mixed_pool_items = []
            for c in components:
                real_name = name_map[c]
                subset = fold_pools[real_name][:limit_per_comp]
                for model in subset:
                    mixed_pool_items.append((model, real_name))

            current_total = len(mixed_pool_items)
            eval_max = min(current_total, exp_cfg.get("max_consensus_size", 20))

            mixed_results = evaluate_mixed_consensus(
                mixed_pool_items, fold_X_tests, y_test, eval_max, n_repeats, np.random.RandomState(seed + fold + 100)
            )

            for r in mixed_results:
                r['Fold'] = fold
                r['Feature Set'] = exp_name
                all_results_df.append(r)

    print_step(3, 4, "汇总结果")
    df = pd.DataFrame(all_results_df)

    df_mean = df.groupby(['Feature Set', 'Metric', 'Consensus Size'])['Value'].agg(['mean', 'std']).reset_index()
    df_mean.columns = ['Feature Set', 'Metric', 'Consensus Size', 'Mean', 'Std']

    df.to_csv(output_dir / "consensus_raw_results.csv", index=False)
    df_mean.to_csv(output_dir / "consensus_summary_results.csv", index=False)

    print_step(4, 4, "可视化")
    if out_cfg.get("generate_plots", True):
        plot_consensus_curves(df_mean, output_dir)

    print_success(f"完成，输出目录: {output_dir}")

def plot_consensus_curves(df: pd.DataFrame, output_dir: Path):
    """共识曲线（两组子图）。"""
    set_nature_style()

    target_metrics = ['Accuracy', 'Sensitivity', 'Specificity']

    colors_map = {
        "Fusion (Feat)": NATURE_COLORS['red'],
        "D": NATURE_COLORS['green'],
        "T": NATURE_COLORS['blue'],
        "S": '#5F9EA0',
        "S+T+D (Mixed)": NATURE_COLORS['red'],
        "S+T (Mixed)":   '#5B396B',
        "S+D (Mixed)":   '#D95F02',
        "T+D (Mixed)":   '#2166AC'
    }

    label_map = {
        "Fusion (Feat)": "Fusion (Feat)",
        "D": "D",
        "T": "T",
        "S": "G",
        "S+T+D (Mixed)": "G+T+D",
        "S+T (Mixed)": "G+T",
        "S+D (Mixed)": "G+D",
        "T+D (Mixed)": "T+D",
    }

    plot_configs = [
        {
            "name": "Single_Feature_Ensembles",
            "filename": "Fig_Consensus_Single",
            "sets": ["Fusion (Feat)", "D", "T", "S"],
            "title_suffix": "(Single Feature Pools)"
        },
        {
            "name": "Mixed_Expert_Ensembles",
            "filename": "Fig_Consensus_Mixed",
            "sets": ["S+T+D (Mixed)", "S+T (Mixed)", "S+D (Mixed)", "T+D (Mixed)"],
            "title_suffix": "(Model-Level Fusion)"
        }
    ]

    for cfg in plot_configs:
        console.print(f"[cyan]生成图表 {cfg['filename']}[/cyan]")

        relevant_sets = [s for s in cfg['sets'] if s in df['Feature Set'].unique()]
        if not relevant_sets:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for i, metric in enumerate(target_metrics):
            if i >= len(axes):
                break
            ax = axes[i]

            df_m = df[df['Metric'] == metric]

            for fs in relevant_sets:
                data = df_m[df_m['Feature Set'] == fs]
                if data.empty:
                    continue

                data = data.sort_values('Consensus Size')

                x = data['Consensus Size']
                y = data['Mean']
                std = data['Std']

                color = colors_map.get(fs, 'gray')
                display_label = label_map.get(fs, fs)

                ax.plot(x, y, label=display_label, color=color, linewidth=2.0)
                ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, edgecolor=None)

            ax.set_title(metric, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Consensus Size', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['bottom'].set_linewidth(1.0)
            ax.tick_params(width=1.0)
            ax.grid(False)

            if i == 0:
                ax.legend(fontsize=9, loc='lower right', frameon=False)

        plt.tight_layout()
        plt.savefig(output_dir / f"{cfg['filename']}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_dir / f"{cfg['filename']}.pdf", bbox_inches='tight', facecolor='white')
        print_success(f"已保存: {cfg['filename']}")
