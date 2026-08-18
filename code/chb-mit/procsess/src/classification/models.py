# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""预置 sklearn 分类器；按名称返回实例。"""

from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def get_models(random_seed: int = 42) -> Dict[str, Any]:
    """名称到分类器实例。"""
    return {
        "SVM": SVC(kernel='rbf', probability=True, random_state=random_seed),
        "RF": RandomForestClassifier(n_estimators=100, random_state=random_seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LR": LogisticRegression(max_iter=1000, random_state=random_seed),
        "GBDT": GradientBoostingClassifier(random_state=random_seed)
    }

def get_model(name: str, random_seed: int = 42) -> Any:
    """按名称返回单个分类器实例。"""
    models = get_models(random_seed)
    if name in models:
        return models[name]
    else:
        raise ValueError(f"Unknown model: {name}")


def build_model(name: str, random_seed: int = 42, params: Optional[Dict[str, Any]] = None) -> Any:
    """按名称构造分类器，并可注入一组已调优参数。"""
    model = get_model(name, random_seed=random_seed)
    if params:
        model.set_params(**params)
    return model
