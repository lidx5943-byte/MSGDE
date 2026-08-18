# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""分类器与交叉验证。"""

from .models import get_models, get_model
from .evaluation import evaluate_model_cv

__all__ = ["get_models", "get_model", "evaluate_model_cv"]
