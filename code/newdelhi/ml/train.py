#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习训练脚本
支持KNN、随机森林、GBDT、SVM模型训练
使用n折交叉验证和KNN参数调优
"""

import os
import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
import shutil

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 设置matplotlib中文字体，避免中文显示警告
# 过滤matplotlib的中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 尝试设置中文字体
try:
    import matplotlib.font_manager as fm
    # 常见的中文字体列表
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
                     'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'STHeiti', 
                     'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
except Exception:
    pass

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理None值：将字符串"None"或YAML的null转换为Python的None
    def convert_none(obj):
        if isinstance(obj, dict):
            return {k: convert_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_none(item) for item in obj]
        elif obj == "None" or obj is None:
            return None
        else:
            return obj
    
    config = convert_none(config)
    return config


def extract_statistical_features(data: np.ndarray) -> np.ndarray:
    """
    从x方向提取统计特征
    
    参数:
        data: 形状为 (100, 1001, 3) 的数据
        
    返回:
        形状为 (100, 6) 的特征矩阵
        特征顺序: [平均值, 最大值, 最小值, 中位数, 方差, 标准差]
    """
    n_samples = data.shape[0]
    features = np.zeros((n_samples, 6))
    
    # 提取x方向的数据 (第三个维度索引为0)
    x_data = data[:, :, 0]  # 形状: (100, 1001)
    
    for i in range(n_samples):
        sample = x_data[i, :]  # 形状: (1001,)
        features[i, 0] = np.mean(sample)      # 平均值
        features[i, 1] = np.max(sample)       # 最大值
        features[i, 2] = np.min(sample)       # 最小值
        features[i, 3] = np.median(sample)    # 中位数
        features[i, 4] = np.var(sample)       # 方差
        features[i, 5] = np.std(sample)       # 标准差
    
    return features


def preprocess_data(data_path: str, label_path: str, scale_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据预处理
    
    参数:
        data_path: 输入数据路径，形状为 (10, 100, 1001, 3)
        label_path: 标签数据路径，形状为 (100,)
        scale_k: 选择的尺度K (0-9)
        
    返回:
        X: 特征矩阵，形状为 (100, 6)
        y: 标签向量，形状为 (100,)
    """
    # 加载数据
    data = np.load(data_path)  # 形状: (10, 100, 1001, 3)
    labels = np.load(label_path)  # 形状: (100,)
    
    # 提取指定尺度K的数据
    scale_data = data[scale_k, :, :, :]  # 形状: (100, 1001, 3)
    
    # 提取统计特征
    X = extract_statistical_features(scale_data)  # 形状: (100, 6)
    
    return X, labels


def tune_knn_k(X: np.ndarray, y: np.ndarray, k_range: List[int], 
               cv: StratifiedKFold) -> Tuple[int, float]:
    """
    KNN的K值调参
    
    参数:
        X: 特征矩阵
        y: 标签向量
        k_range: K值范围
        cv: 交叉验证对象
        
    返回:
        best_k: 最佳K值
        best_score: 最佳得分
    """
    best_k = k_range[0]
    best_score = 0.0
    
    print("正在调优KNN的K值...")
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        print(f"  K={k}: 平均准确率 = {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    
    print(f"最佳K值: {best_k}, 最佳得分: {best_score:.4f}\n")
    return best_k, best_score


def calculate_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    计算敏感性和特异性指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        包含敏感性和特异性的字典
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    # 存储每个类别的敏感性和特异性
    sensitivities = []
    specificities = []
    
    for i in range(n_classes):
        # TP: 正确预测为类别i的样本数
        tp = cm[i, i]
        # FN: 实际是类别i但被预测为其他类别的样本数
        fn = cm[i, :].sum() - tp
        # FP: 实际不是类别i但被预测为类别i的样本数
        fp = cm[:, i].sum() - tp
        # TN: 正确预测为不是类别i的样本数
        tn = cm.sum() - tp - fn - fp
        
        # 敏感性 (Sensitivity/Recall/TPR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivities.append(sensitivity)
        
        # 特异性 (Specificity/TNR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    # 计算宏平均和加权平均
    sensitivity_macro = np.mean(sensitivities)
    specificity_macro = np.mean(specificities)
    
    # 加权平均（按类别样本数加权）
    class_counts = np.bincount(y_true)
    weights = class_counts / class_counts.sum()
    sensitivity_weighted = np.average(sensitivities, weights=weights)
    specificity_weighted = np.average(specificities, weights=weights)
    
    return {
        'sensitivity_per_class': sensitivities,
        'specificity_per_class': specificities,
        'sensitivity_macro': sensitivity_macro,
        'specificity_macro': specificity_macro,
        'sensitivity_weighted': sensitivity_weighted,
        'specificity_weighted': specificity_weighted
    }


def train_and_evaluate(model, model_name: str, X: np.ndarray, y: np.ndarray,
                       cv: StratifiedKFold) -> Dict:
    """
    训练模型并进行交叉验证评估
    
    参数:
        model: 模型对象
        model_name: 模型名称
        X: 特征矩阵
        y: 标签向量
        cv: 交叉验证对象
        
    返回:
        包含评估结果的字典
    """
    print(f"正在训练 {model_name}...")
    
    # 交叉验证 - 获取准确率
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # 交叉验证 - 获取预测结果（用于计算所有指标）
    # 这确保每个样本的预测都是在它作为测试集时得到的，避免过拟合
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    # 使用交叉验证的预测结果计算所有指标
    accuracy = accuracy_score(y, y_pred_cv)
    precision = precision_score(y, y_pred_cv, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred_cv, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred_cv, average='weighted', zero_division=0)
    
    # 混淆矩阵（基于交叉验证预测）
    cm = confusion_matrix(y, y_pred_cv)
    
    # 分类报告（基于交叉验证预测）
    report = classification_report(y, y_pred_cv, zero_division=0)
    
    # 计算敏感性和特异性（基于交叉验证预测）
    sens_spec = calculate_sensitivity_specificity(y, y_pred_cv)
    
    # 在整个数据集上训练最终模型（用于保存模型，但不用于评估）
    model.fit(X, y)
    
    results = {
        'model_name': model_name,
        'cv_mean_accuracy': cv_scores.mean(),
        'cv_std_accuracy': cv_scores.std(),
        'cv_scores': cv_scores.tolist(),
        'accuracy': accuracy,  # 现在这是交叉验证的准确率，与cv_mean_accuracy应该一致
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity_macro': sens_spec['sensitivity_macro'],
        'sensitivity_weighted': sens_spec['sensitivity_weighted'],
        'specificity_macro': sens_spec['specificity_macro'],
        'specificity_weighted': sens_spec['specificity_weighted'],
        'sensitivity_per_class': sens_spec['sensitivity_per_class'],
        'specificity_per_class': sens_spec['specificity_per_class'],
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    print(f"  {model_name} 交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  交叉验证准确率（直接计算）: {accuracy:.4f}")
    print(f"  敏感性(加权): {sens_spec['sensitivity_weighted']:.4f}, 特异性(加权): {sens_spec['specificity_weighted']:.4f}\n")
    
    return results, model, cm


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results: Dict, output_dir: Path, X: np.ndarray, y: np.ndarray, 
                 config: Dict, config_path: str):
    """保存所有结果"""
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / 'img'
    img_dir.mkdir(exist_ok=True)
    
    # 保存特征和标签
    np.save(output_dir / 'X_feature.npy', X)
    np.save(output_dir / 'y_label.npy', y)
    
    # 保存每个模型的结果
    all_results = {}
    report_lines = []
    
    for result in results:
        model_name = result['model_name']
        all_results[model_name] = {
            'cv_mean_accuracy': result['cv_mean_accuracy'],
            'cv_std_accuracy': result['cv_std_accuracy'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'sensitivity_macro': result['sensitivity_macro'],
            'sensitivity_weighted': result['sensitivity_weighted'],
            'specificity_macro': result['specificity_macro'],
            'specificity_weighted': result['specificity_weighted'],
            'sensitivity_per_class': result['sensitivity_per_class'],
            'specificity_per_class': result['specificity_per_class']
        }
        
        # 保存混淆矩阵图片
        cm = np.array(result['confusion_matrix'])
        plot_confusion_matrix(cm, model_name, img_dir / f'confusion_matrix_{model_name}.png')
        
        # 添加到报告
        report_lines.append(f"\n{'='*60}\n")
        report_lines.append(f"模型: {model_name}\n")
        report_lines.append(f"{'='*60}\n")
        report_lines.append(f"交叉验证平均准确率: {result['cv_mean_accuracy']:.4f} (+/- {result['cv_std_accuracy'] * 2:.4f})\n")
        report_lines.append(f"准确率: {result['accuracy']:.4f}\n")
        report_lines.append(f"精确率: {result['precision']:.4f}\n")
        report_lines.append(f"召回率: {result['recall']:.4f}\n")
        report_lines.append(f"F1分数: {result['f1_score']:.4f}\n")
        report_lines.append(f"敏感性(宏平均): {result['sensitivity_macro']:.4f}\n")
        report_lines.append(f"敏感性(加权平均): {result['sensitivity_weighted']:.4f}\n")
        report_lines.append(f"特异性(宏平均): {result['specificity_macro']:.4f}\n")
        report_lines.append(f"特异性(加权平均): {result['specificity_weighted']:.4f}\n")
        
        # 添加每个类别的敏感性和特异性
        if len(result['sensitivity_per_class']) > 0:
            report_lines.append(f"\n各类别敏感性: {[f'{s:.4f}' for s in result['sensitivity_per_class']]}\n")
            report_lines.append(f"各类别特异性: {[f'{s:.4f}' for s in result['specificity_per_class']]}\n")
        
        report_lines.append(f"\n分类报告:\n{result['classification_report']}\n")
    
    # 保存JSON结果
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存文本报告
    with open(output_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(f"实验名称: {config['experiment_name']}\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"交叉验证折数: {config['n_folds']}\n")
        f.write(f"选择的尺度K: {config['scale_k']}\n")
        f.write(f"特征数量: {X.shape[1]}\n")
        f.write(f"样本数量: {X.shape[0]}\n")
        f.write("".join(report_lines))
    
    # 复制配置文件
    shutil.copy(config_path, output_dir / 'config.yaml')
    
    print(f"所有结果已保存到: {output_dir}")


def main():
    """主函数"""
    # 加载配置
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output_path']) / f"{config['experiment_name']}_{timestamp}"
    
    print("="*60)
    print("机器学习训练脚本")
    print("="*60)
    print(f"实验名称: {config['experiment_name']}")
    print(f"输出目录: {output_dir}")
    print(f"交叉验证折数: {config['n_folds']}")
    print(f"选择的尺度K: {config['scale_k']}")
    print("="*60)
    print()
    
    # 数据预处理
    print("正在加载和预处理数据...")
    X, y = preprocess_data(
        config['input_data'],
        config['label'],
        config['scale_k']
    )
    print(f"特征形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print()
    
    # 创建交叉验证对象
    cv = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    
    # 存储所有结果
    all_results = []
    
    # 1. KNN (带K值调优)
    print("="*60)
    print("KNN模型")
    print("="*60)
    best_k, best_k_score = tune_knn_k(X, y, config['knn_k_range'], cv)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn_result, knn_model, knn_cm = train_and_evaluate(knn, 'KNN', X, y, cv)
    knn_result['best_k'] = best_k
    all_results.append(knn_result)
    
    # 2. 随机森林
    print("="*60)
    print("随机森林模型")
    print("="*60)
    max_depth_rf = config['random_forest']['max_depth']
    if max_depth_rf == "None" or (isinstance(max_depth_rf, str) and max_depth_rf.lower() == 'none'):
        max_depth_rf = None
    rf = RandomForestClassifier(
        n_estimators=config['random_forest']['n_estimators'],
        max_depth=max_depth_rf,
        random_state=config['random_forest']['random_state']
    )
    rf_result, rf_model, rf_cm = train_and_evaluate(rf, 'RandomForest', X, y, cv)
    all_results.append(rf_result)
    
    # 3. GBDT
    print("="*60)
    print("GBDT模型")
    print("="*60)
    gbdt = GradientBoostingClassifier(
        n_estimators=config['gbdt']['n_estimators'],
        learning_rate=config['gbdt']['learning_rate'],
        max_depth=config['gbdt']['max_depth'],
        random_state=config['gbdt']['random_state']
    )
    gbdt_result, gbdt_model, gbdt_cm = train_and_evaluate(gbdt, 'GBDT', X, y, cv)
    all_results.append(gbdt_result)
    
    # 4. SVM
    print("="*60)
    print("SVM模型")
    print("="*60)
    svm = SVC(
        C=config['svm']['C'],
        kernel=config['svm']['kernel'],
        gamma=config['svm']['gamma'],
        random_state=config['svm']['random_state']
    )
    svm_result, svm_model, svm_cm = train_and_evaluate(svm, 'SVM', X, y, cv)
    all_results.append(svm_result)
    
    # 保存结果
    print("="*60)
    print("保存结果")
    print("="*60)
    save_results(all_results, output_dir, X, y, config, config_path)
    
    # 打印总结
    print("\n" + "="*60)
    print("训练完成！结果总结:")
    print("="*60)
    print(f"{'模型':<15s} {'CV准确率':<15s} {'敏感性':<12s} {'特异性':<12s}")
    print("-"*60)
    for result in all_results:
        print(f"{result['model_name']:<15s} {result['cv_mean_accuracy']:.4f}±{result['cv_std_accuracy']*2:.4f}  "
              f"{result['sensitivity_weighted']:.4f}      {result['specificity_weighted']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()


