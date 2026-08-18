# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""交叉验证评估与指标。"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for k in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[k, :]) + np.sum(cm[:, k]) - cm[k, k])
        fp = np.sum(cm[:, k]) - cm[k, k]
        if tn + fp == 0:
            specs.append(0.0)
        else:
            specs.append(tn / (tn + fp))
    return float(np.mean(specs)) if specs else 0.0


def _compute_auc(model: Any, y_true: np.ndarray, X_eval: np.ndarray, y_all: np.ndarray) -> float:
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)
            n_classes_y = len(np.unique(y_all))

            if n_classes_y == 2:
                prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.ravel()
                return float(roc_auc_score(y_true, prob))
            if y_prob.shape[1] == n_classes_y:
                return float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
    except Exception:
        return 0.0
    return 0.0


def _summarize_metrics(
    acc_scores: list[float],
    prec_scores: list[float],
    rec_scores: list[float],
    spec_scores: list[float],
    f1_scores: list[float],
    auc_scores: list[float],
    model_name: str,
    feature_set_name: str,
) -> Dict[str, float]:
    return {
        "Feature Set": feature_set_name,
        "Model": model_name,
        "Accuracy": float(np.mean(acc_scores)) if acc_scores else 0.0,
        "Precision": float(np.mean(prec_scores)) if prec_scores else 0.0,
        "Sensitivity": float(np.mean(rec_scores)) if rec_scores else 0.0,
        "Specificity": float(np.mean(spec_scores)) if spec_scores else 0.0,
        "F1-Score": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "AUC": float(np.mean(auc_scores)) if auc_scores else 0.0,
    }

def evaluate_model_cv(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    cv_splits: int = 5,
    random_seed: int = 42,
    model_name: str = "Model",
    feature_set_name: str = "FeatureSet"
) -> Tuple[Dict[str, float], Any]:
    """交叉验证；返回指标字典与 (y_true, y_prob) 供 ROC。"""
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    spec_scores = []
    auc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0)) 
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        spec_scores.append(_compute_specificity(y_test, y_pred))
        auc_scores.append(_compute_auc(fold_model, y_test, X_test, y))

    roc_info = None
    try:
        if hasattr(model, "predict_proba"):
            from sklearn.pipeline import make_pipeline
            clf_pipe = make_pipeline(StandardScaler(), clone(model))
            y_prob_all = cross_val_predict(clf_pipe, X, y, cv=cv, method='predict_proba')
            roc_info = (y, y_prob_all)
    except Exception:
        pass

    metrics = _summarize_metrics(
        acc_scores, prec_scores, rec_scores, spec_scores, f1_scores, auc_scores, model_name, feature_set_name
    )
    return metrics, roc_info


def evaluate_model_cv_transfer(
    model: Any,
    X_train_domain: np.ndarray,
    X_test_domain: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    random_seed: int = 42,
    model_name: str = "Model",
    feature_set_name: str = "FeatureSet",
) -> Tuple[Dict[str, float], Any]:
    """在干净域训练、目标域测试的配对交叉验证。"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    spec_scores = []
    auc_scores = []

    prob_chunks = []
    truth_chunks = []

    for train_idx, test_idx in cv.split(X_train_domain, y):
        X_train = X_train_domain[train_idx]
        X_test = X_test_domain[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        spec_scores.append(_compute_specificity(y_test, y_pred))
        auc_scores.append(_compute_auc(fold_model, y_test, X_test, y))

        if hasattr(fold_model, "predict_proba"):
            try:
                prob_chunks.append(fold_model.predict_proba(X_test))
                truth_chunks.append(y_test)
            except Exception:
                pass

    roc_info = None
    if prob_chunks and truth_chunks:
        try:
            roc_info = (np.concatenate(truth_chunks, axis=0), np.concatenate(prob_chunks, axis=0))
        except Exception:
            roc_info = None

    metrics = _summarize_metrics(
        acc_scores, prec_scores, rec_scores, spec_scores, f1_scores, auc_scores, model_name, feature_set_name
    )
    return metrics, roc_info
