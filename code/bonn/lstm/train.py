#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM/RNN 训练脚本
支持 SimpleRNN 和 LSTM 模型训练
使用n折交叉验证
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
import shutil
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical

import random

warnings.filterwarnings('ignore')


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理None值
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


def set_random_seeds(seed: int):
    """设置所有随机数种子"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_data(data_path: str, label_path: str, scale_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载数据
    
    参数:
        data_path: 输入数据路径
            - 如果数据是多尺度的，形状为 (n_scales, n_samples, timesteps, features)，例如 (10, 200, 1001, 3)
            - 如果数据是单尺度的，形状为 (n_samples, timesteps, features)，例如 (200, 200, 3)
        label_path: 标签数据路径，支持 .xlsx 或 .npy 格式，形状为 (n_samples,)
        scale_k: 选择的尺度K (0到n_scales-1)，仅当输入数据是多尺度时使用
        
    返回:
        X: 特征数据，形状为 (n_samples, timesteps, features)
        y: 标签数据，形状为 (n_samples,)
    """
    # 加载特征数据
    X = np.load(data_path)
    print(f"原始数据形状: {X.shape}")
    
    # 判断数据是否为多尺度格式（第一个维度是尺度维度）
    # 如果数据是4维的，认为是多尺度数据
    if len(X.shape) == 4:
        if scale_k is None:
            raise ValueError("输入数据是多尺度格式，必须指定 scale_k 参数")
        if scale_k < 0 or scale_k >= X.shape[0]:
            raise ValueError(f"scale_k ({scale_k}) 超出范围 [0, {X.shape[0]-1}]")
        # 提取指定尺度K的数据
        X = X[scale_k, :, :, :]  # 形状: (n_samples, timesteps, features)
        print(f"选择尺度 {scale_k}，提取后的数据形状: {X.shape}")
    elif len(X.shape) == 3:
        # 单尺度数据，直接使用
        if scale_k is not None:
            print(f"警告: 输入数据是单尺度格式，scale_k 参数将被忽略")
        print(f"数据形状: {X.shape}")
    else:
        raise ValueError(f"不支持的数据维度: {len(X.shape)}，期望3维或4维")
    
    # 加载标签数据
    if label_path.endswith('.xlsx') or label_path.endswith('.xls'):
        df = pd.read_excel(label_path)
        if 'label' in df.columns:
            y = df['label'].values
        else:
            # 如果没有 'label' 列，使用第一列
            y = df.iloc[:, 0].values
    elif label_path.endswith('.npy'):
        y = np.load(label_path)
    else:
        raise ValueError(f"不支持的标签文件格式: {label_path}")
    
    print(f"标签形状: {y.shape}")
    
    # 确保标签是整数类型
    y = y.astype(int)
    
    # 验证数据维度匹配
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"数据样本数 ({X.shape[0]}) 与标签数 ({y.shape[0]}) 不匹配")
    
    return X, y


def preprocess_data(data: np.ndarray, method: str = "standardize") -> np.ndarray:
    """
    数据预处理
    
    参数:
        data: 输入数据，形状为 (n_samples, timesteps, features)
        method: 预处理方法，"standardize", "normalize" 或 "none"
        
    返回:
        预处理后的数据
    """
    if method == "none":
        return data
    
    # 将数据展平为 (n_samples * timesteps, features) 进行预处理
    original_shape = data.shape
    data_reshaped = data.reshape(-1, data.shape[-1])
    
    if method == "standardize":
        scaler = StandardScaler()
        data_processed = scaler.fit_transform(data_reshaped)
    elif method == "normalize":
        scaler = MinMaxScaler()
        data_processed = scaler.fit_transform(data_reshaped)
    else:
        raise ValueError(f"未知的预处理方法: {method}")
    
    # 恢复原始形状
    data_processed = data_processed.reshape(original_shape)
    
    return data_processed


def create_rnn_model(config: Dict, input_shape: Tuple) -> Sequential:
    """
    创建RNN模型
    
    参数:
        config: RNN配置字典
        input_shape: 输入形状 (timesteps, features)
        
    返回:
        Keras模型
    """
    model = Sequential()
    rnn_config = config['rnn']
    
    # 第一层RNN
    if rnn_config['model_type'] == 'SimpleRNN':
        model.add(SimpleRNN(
            rnn_config['units_first'],
            activation=rnn_config['activation'],
            input_shape=input_shape,
            return_sequences=rnn_config['use_stacked']
        ))
    elif rnn_config['model_type'] == 'LSTM':
        model.add(LSTM(
            rnn_config['units_first'],
            activation=rnn_config['activation'],
            input_shape=input_shape,
            return_sequences=rnn_config['use_stacked']
        ))
    else:
        raise ValueError(f"未知的RNN类型: {rnn_config['model_type']}")
    
    model.add(Dropout(rnn_config['dropout_rate']))
    
    # 第二层RNN（如果使用堆叠）
    if rnn_config['use_stacked']:
        if rnn_config['model_type'] == 'SimpleRNN':
            model.add(SimpleRNN(
                rnn_config['units_second'],
                activation=rnn_config['activation'],
                return_sequences=False
            ))
        else:
            model.add(LSTM(
                rnn_config['units_second'],
                activation=rnn_config['activation'],
                return_sequences=False
            ))
        model.add(Dropout(rnn_config['dropout_rate']))
    
    # 全连接层
    model.add(Dense(rnn_config['dense_units'], activation=rnn_config['dense_activation']))
    model.add(Dropout(rnn_config['dropout_rate']))
    
    # 输出层
    model.add(Dense(rnn_config['num_classes'], activation='softmax'))
    
    # 编译模型
    train_config = config['training']
    optimizer = Adam(learning_rate=train_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(config: Dict, input_shape: Tuple) -> Sequential:
    """
    创建LSTM模型
    
    参数:
        config: LSTM配置字典
        input_shape: 输入形状 (timesteps, features)
        
    返回:
        Keras模型
    """
    model = Sequential()
    lstm_config = config['lstm']
    
    # 第一层LSTM
    model.add(LSTM(
        lstm_config['units_first'],
        activation=lstm_config['activation'],
        input_shape=input_shape,
        return_sequences=lstm_config['use_stacked']
    ))
    model.add(Dropout(lstm_config['dropout_rate']))
    
    # 第二层LSTM（如果使用堆叠）
    if lstm_config['use_stacked']:
        model.add(LSTM(
            lstm_config['units_second'],
            activation=lstm_config['activation'],
            return_sequences=False
        ))
        model.add(Dropout(lstm_config['dropout_rate']))
    
    # 全连接层
    model.add(Dense(lstm_config['dense_units'], activation=lstm_config['dense_activation']))
    model.add(Dropout(lstm_config['dropout_rate']))
    
    # 输出层
    model.add(Dense(lstm_config['num_classes'], activation='softmax'))
    
    # 编译模型
    train_config = config['training']
    optimizer = Adam(learning_rate=train_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算特异性（多分类的宏平均）
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        特异性（宏平均）
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fp - (cm[i, :].sum() - tp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    return np.mean(specificities)


def train_and_evaluate(model, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       config: Dict, fold: int) -> Dict:
    """
    训练模型并评估
    
    参数:
        model: Keras模型
        X_train: 训练数据
        y_train: 训练标签（one-hot编码）
        X_test: 测试数据
        y_test: 测试标签（one-hot编码）
        config: 配置字典
        fold: 当前折数
        
    返回:
        包含评估结果的字典
    """
    train_config = config['training']
    
    # 创建回调函数
    callbacks = []
    
    # 早停机制
    early_stop = EarlyStopping(
        monitor=train_config['early_stopping']['monitor'],
        patience=train_config['early_stopping']['patience'],
        restore_best_weights=train_config['early_stopping']['restore_best_weights'],
        verbose=0
    )
    callbacks.append(early_stop)
    
    # 学习率调度
    if train_config['learning_rate_scheduler']['enabled']:
        def lr_scheduler(epoch, lr):
            decay_epoch = train_config['learning_rate_scheduler']['decay_epoch']
            decay_factor = train_config['learning_rate_scheduler']['decay_factor']
            if epoch > decay_epoch:
                return lr * decay_factor
            return lr
        
        lr_schedule = LearningRateScheduler(lr_scheduler, verbose=0)
        callbacks.append(lr_schedule)
    
    # 训练模型
    model.fit(
        X_train, y_train,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0
    )
    
    # 预测
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred_classes)
    ppv = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
    sensitivity = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
    specificity = calculate_specificity(y_true, y_pred_classes)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    report = classification_report(y_true, y_pred_classes, zero_division=0)
    
    return {
        'fold': fold,
        'accuracy': accuracy,
        'ppv': ppv,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def save_results(results: Dict, output_dir: Path, config: Dict, config_path: str):
    """保存所有结果"""
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个模型的结果
    all_results = {}
    report_lines = []
    
    for model_name, model_results in results.items():
        # 计算平均值和标准差
        accuracies = [r['accuracy'] for r in model_results]
        ppvs = [r['ppv'] for r in model_results]
        sensitivities = [r['sensitivity'] for r in model_results]
        specificities = [r['specificity'] for r in model_results]
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_ppv = np.mean(ppvs)
        std_ppv = np.std(ppvs)
        avg_sensitivity = np.mean(sensitivities)
        std_sensitivity = np.std(sensitivities)
        avg_specificity = np.mean(specificities)
        std_specificity = np.std(specificities)
        
        all_results[model_name] = {
            'avg_accuracy': float(avg_accuracy),
            'std_accuracy': float(std_accuracy),
            'avg_ppv': float(avg_ppv),
            'std_ppv': float(std_ppv),
            'avg_sensitivity': float(avg_sensitivity),
            'std_sensitivity': float(std_sensitivity),
            'avg_specificity': float(avg_specificity),
            'std_specificity': float(std_specificity),
            'fold_results': model_results
        }
        
        # 添加到报告
        report_lines.append(f"\n{'='*60}\n")
        report_lines.append(f"模型: {model_name}\n")
        report_lines.append(f"{'='*60}\n")
        report_lines.append(f"平均准确率: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        report_lines.append(f"平均PPV: {avg_ppv:.4f} ± {std_ppv:.4f}\n")
        report_lines.append(f"平均敏感性: {avg_sensitivity:.4f} ± {std_sensitivity:.4f}\n")
        report_lines.append(f"平均特异性: {avg_specificity:.4f} ± {std_specificity:.4f}\n")
        
        # 添加每折的详细结果
        report_lines.append(f"\n各折详细结果:\n")
        for result in model_results:
            report_lines.append(f"  折 {result['fold']}: "
                              f"准确率={result['accuracy']:.4f}, "
                              f"PPV={result['ppv']:.4f}, "
                              f"敏感性={result['sensitivity']:.4f}, "
                              f"特异性={result['specificity']:.4f}\n")
    
    # 保存JSON结果
    with open(output_dir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存文本报告
    with open(output_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(f"实验名称: {config['experiment_name']}\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"交叉验证折数: {config['n_folds']}\n")
        if 'scale_k' in config:
            f.write(f"选择的尺度K: {config['scale_k']}\n")
        f.write(f"数据预处理: {config['preprocessing']}\n")
        f.write("".join(report_lines))
    
    # 复制配置文件
    shutil.copy(config_path, output_dir / 'config.yaml')
    
    print(f"所有结果已保存到: {output_dir}")


def main():
    """主函数"""
    # 加载配置
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # 设置随机数种子
    set_random_seeds(config['random_state'])
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['output_path']) / f"{config['experiment_name']}_{timestamp}"
    
    print("="*60)
    print("LSTM/RNN 训练脚本")
    print("="*60)
    print(f"实验名称: {config['experiment_name']}")
    print(f"输出目录: {output_dir}")
    print(f"交叉验证折数: {config['n_folds']}")
    if 'scale_k' in config:
        print(f"选择的尺度K: {config['scale_k']}")
    print(f"数据预处理: {config['preprocessing']}")
    print("="*60)
    print()
    
    # 加载数据
    print("正在加载数据...")
    scale_k = config.get('scale_k', None)
    X, y = load_data(config['input_data'], config['label'], scale_k=scale_k)
    print(f"最终数据形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"类别数: {len(np.unique(y))}")
    print()
    
    # 数据预处理
    print(f"正在预处理数据（方法: {config['preprocessing']}）...")
    X = preprocess_data(X, config['preprocessing'])
    print()
    
    # 将标签转换为one-hot编码
    num_classes = config['rnn']['num_classes']  # 假设RNN和LSTM的类别数相同
    y_one_hot = to_categorical(y, num_classes=num_classes)
    
    # 创建交叉验证对象
    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_state'])
    
    # 存储所有结果
    all_results = {
        'RNN': [],
        'LSTM': []
    }
    
    # 训练RNN模型
    print("="*60)
    print("训练RNN模型")
    print("="*60)
    input_shape = (X.shape[1], X.shape[2])
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"训练第 {fold}/{config['n_folds']} 折...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
        
        # 创建模型
        model = create_rnn_model(config, input_shape)
        
        # 训练和评估
        result = train_and_evaluate(model, X_train, y_train, X_test, y_test, config, fold)
        all_results['RNN'].append(result)
        
        print(f"  准确率: {result['accuracy']:.4f}, "
              f"PPV: {result['ppv']:.4f}, "
              f"敏感性: {result['sensitivity']:.4f}, "
              f"特异性: {result['specificity']:.4f}")
    
    print()
    
    # 训练LSTM模型
    print("="*60)
    print("训练LSTM模型")
    print("="*60)
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"训练第 {fold}/{config['n_folds']} 折...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
        
        # 创建模型
        model = create_lstm_model(config, input_shape)
        
        # 训练和评估
        result = train_and_evaluate(model, X_train, y_train, X_test, y_test, config, fold)
        all_results['LSTM'].append(result)
        
        print(f"  准确率: {result['accuracy']:.4f}, "
              f"PPV: {result['ppv']:.4f}, "
              f"敏感性: {result['sensitivity']:.4f}, "
              f"特异性: {result['specificity']:.4f}")
    
    print()
    
    # 保存结果
    print("="*60)
    print("保存结果")
    print("="*60)
    save_results(all_results, output_dir, config, config_path)
    
    # 打印总结
    print("\n" + "="*60)
    print("训练完成！结果总结:")
    print("="*60)
    print(f"{'模型':<10s} {'准确率':<15s} {'PPV':<15s} {'敏感性':<15s} {'特异性':<15s}")
    print("-"*60)
    
    for model_name, model_results in all_results.items():
        accuracies = [r['accuracy'] for r in model_results]
        ppvs = [r['ppv'] for r in model_results]
        sensitivities = [r['sensitivity'] for r in model_results]
        specificities = [r['specificity'] for r in model_results]
        
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_ppv = np.mean(ppvs)
        std_ppv = np.std(ppvs)
        avg_sens = np.mean(sensitivities)
        std_sens = np.std(sensitivities)
        avg_spec = np.mean(specificities)
        std_spec = np.std(specificities)
        
        print(f"{model_name:<10s} {avg_acc:.4f}±{std_acc:.4f}  "
              f"{avg_ppv:.4f}±{std_ppv:.4f}  "
              f"{avg_sens:.4f}±{std_sens:.4f}  "
              f"{avg_spec:.4f}±{std_spec:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

