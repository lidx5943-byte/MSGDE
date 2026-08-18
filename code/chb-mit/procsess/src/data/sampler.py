# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""平衡采样与按患者分层。"""

import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .window_generator import WindowMeta
from ..utils.console import console, print_success, print_warning, print_error, print_info


@dataclass
class SamplingResult:
    """
    采样结果
    
    Attributes:
        sampled_windows: 采样后的窗口列表
        preictal_count: Preictal 样本数
        interictal_count: Interictal 样本数
        patients_sampled: 采样涉及的患者列表
        warnings: 警告消息列表
    """
    sampled_windows: List[WindowMeta]
    preictal_count: int
    interictal_count: int
    patients_sampled: List[str]
    warnings: List[str]


class BalancedSampler:
    """
    平衡采样器
    
    从候选窗口池中进行 1:1 平衡采样，支持：
    - 按患者分层采样
    - 样本数量验证和警告
    - 可复现的随机采样
    """
    
    # 标签常量
    LABEL_INTERICTAL = 0
    LABEL_PREICTAL = 1
    
    def __init__(self, seed: int = 42):
        """
        初始化采样器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        random.seed(seed)
    
    def validate_availability(
        self,
        windows: List[WindowMeta],
        samples_per_label: int
    ) -> Tuple[bool, str, Dict[str, int]]:
        """
        验证样本可用性
        
        Args:
            windows: 候选窗口列表
            samples_per_label: 每个标签请求的样本数
        
        Returns:
            (是否可用, 消息, 统计信息)
        """
        # 统计每类样本数
        preictal_count = sum(1 for w in windows if w.label == self.LABEL_PREICTAL)
        interictal_count = sum(1 for w in windows if w.label == self.LABEL_INTERICTAL)
        
        stats = {
            "preictal_available": preictal_count,
            "interictal_available": interictal_count,
            "total_available": len(windows),
            "requested_per_label": samples_per_label,
        }
        
        if samples_per_label == 0:
            # 使用所有可用样本
            min_count = min(preictal_count, interictal_count)
            if min_count == 0:
                return False, "没有可用样本", stats
            stats["will_use_per_label"] = min_count
            return True, f"将使用平衡采样，每类 {min_count} 个样本", stats
        
        # 检查请求数是否超过可用数
        if samples_per_label > preictal_count:
            return False, f"请求的 preictal 样本数 ({samples_per_label}) 超过可用数 ({preictal_count})", stats
        
        if samples_per_label > interictal_count:
            return False, f"请求的 interictal 样本数 ({samples_per_label}) 超过可用数 ({interictal_count})", stats
        
        stats["will_use_per_label"] = samples_per_label
        return True, f"样本充足，将采样每类 {samples_per_label} 个", stats
    
    def sample(
        self,
        windows: List[WindowMeta],
        samples_per_label: int = 0,
        balance_ratio: float = 1.0,
        stratify_by_patient: bool = True
    ) -> SamplingResult:
        """
        执行平衡采样
        
        Args:
            windows: 候选窗口列表
            samples_per_label: 每个标签的样本数（0 表示使用所有可用样本并平衡）
            balance_ratio: preictal:interictal 比例
            stratify_by_patient: 是否按患者分层采样
        
        Returns:
            SamplingResult 对象
        """
        warnings = []
        
        # 分离两类样本
        preictal_windows = [w for w in windows if w.label == self.LABEL_PREICTAL]
        interictal_windows = [w for w in windows if w.label == self.LABEL_INTERICTAL]
        
        print_info(f"候选池: Preictal {len(preictal_windows)}, Interictal {len(interictal_windows)}")
        
        # 统计有数据的患者数
        pre_patients = set(w.patient_id for w in preictal_windows)
        int_patients = set(w.patient_id for w in interictal_windows)
        
        if samples_per_label == 0:
            preictal_target = len(preictal_windows)
        else:
            preictal_target = samples_per_label
            # 确保至少能覆盖所有有数据的患者
            if len(pre_patients) > 0 and preictal_target < len(pre_patients):
                warnings.append(
                    f"请求的 Preictal 样本数 ({preictal_target}) 小于患者数 ({len(pre_patients)})，"
                    f"已自动提升至 {len(pre_patients)} 以确保覆盖每个患者。"
                )
                preictal_target = len(pre_patients)
        
        if samples_per_label == 0:

             if len(preictal_windows) * balance_ratio <= len(interictal_windows):
                 interictal_target = int(len(preictal_windows) * balance_ratio)
             else:
                 min_count = min(len(preictal_windows), int(len(interictal_windows) / balance_ratio))
                 preictal_target = min_count
                 interictal_target = int(min_count * balance_ratio)
        else:
            interictal_target = int(preictal_target * balance_ratio)
            
            # 同样确保 Interictal 覆盖所有患者
            if len(int_patients) > 0 and interictal_target < len(int_patients):
                warnings.append(
                    f"计算出的 Interictal 样本数 ({interictal_target}) 小于患者数 ({len(int_patients)})，"
                    f"已自动提升至 {len(int_patients)} 以确保覆盖每个患者。"
                )
                interictal_target = len(int_patients)
        
        if preictal_target > len(preictal_windows):
            warnings.append(
                f"Preictal 目标数 ({preictal_target}) 超过可用上限 ({len(preictal_windows)})，"
                f"将使用全部可用样本。"
            )
            preictal_target = len(preictal_windows)
            new_interictal = int(preictal_target * balance_ratio)
            if new_interictal >= len(int_patients):
                interictal_target = new_interictal
        
        if interictal_target > len(interictal_windows):
            warnings.append(
                f"Interictal 目标数 ({interictal_target}) 超过可用上限 ({len(interictal_windows)})，"
                f"将使用全部可用样本。"
            )
            interictal_target = len(interictal_windows)
        
        # 打印警告
        for warning in warnings:
            print_warning(warning)
        
        # 执行采样
        if stratify_by_patient:
            sampled_preictal = self._stratified_sample(preictal_windows, preictal_target)
            sampled_interictal = self._stratified_sample(interictal_windows, interictal_target)
        else:
            sampled_preictal = random.sample(preictal_windows, preictal_target)
            sampled_interictal = random.sample(interictal_windows, interictal_target)
        
        # 合并结果
        sampled_windows = sampled_preictal + sampled_interictal
        
        # 打乱顺序
        random.shuffle(sampled_windows)
        
        # 统计采样涉及的患者
        patients = list(set(w.patient_id for w in sampled_windows))
        
        result = SamplingResult(
            sampled_windows=sampled_windows,
            preictal_count=len(sampled_preictal),
            interictal_count=len(sampled_interictal),
            patients_sampled=patients,
            warnings=warnings
        )
        
        print_success(
            f"采样完成: Preictal {result.preictal_count}, "
            f"Interictal {result.interictal_count}, "
            f"来自 {len(patients)} 个患者"
        )
        
        return result
    
    def _stratified_sample(
        self,
        windows: List[WindowMeta],
        target_count: int
    ) -> List[WindowMeta]:
        """
        按患者分层采样（迭代平均分配算法）
        
        目标：
        1. 确保每个有数据的患者至少有一个样本（如果 target_count 允许）。
        2. 尽可能在患者间均匀分配样本。
        3. 当某些患者数据不足时，剩余配额均匀分配给其他数据充足的患者。
        
        Args:
            windows: 候选窗口列表
            target_count: 目标样本数
        
        Returns:
            采样后的窗口列表
        """
        if not windows:
            return []
        if target_count >= len(windows):
            return windows.copy()
        if target_count <= 0:
            return []
            
        # 1. 按患者分组
        patient_windows = defaultdict(list)
        for w in windows:
            patient_windows[w.patient_id].append(w)
        
        patient_ids = sorted(patient_windows.keys())
        # 2. 迭代分配配额
        quotas = {pid: 0 for pid in patient_ids}
        available = {pid: len(patient_windows[pid]) for pid in patient_ids}
        remaining_target = target_count
        
        while remaining_target > 0:
            active_patients = [pid for pid in patient_ids if quotas[pid] < available[pid]]
            
            if not active_patients:
                break 
            
            n_active = len(active_patients)

            step = max(1, remaining_target // n_active)
            
            allocated_in_round = 0
            for pid in active_patients:
                if remaining_target == 0:
                    break
                
                can_take = available[pid] - quotas[pid]
                
                to_take = min(step, can_take, remaining_target)
                
                quotas[pid] += to_take
                remaining_target -= to_take
                allocated_in_round += to_take
            
            if allocated_in_round == 0:
                break 
        
        # 3. 根据配额进行采样
        sampled = []
        for pid in patient_ids:
            count = quotas[pid]
            if count > 0:
                # 随机抽取指定数量
                sampled.extend(random.sample(patient_windows[pid], count))
                
        return sampled


if __name__ == "__main__":
    # 测试采样器
    from .window_generator import WindowMeta
    
    # 创建测试数据
    test_windows = []
    for i in range(100):
        test_windows.append(WindowMeta(
            patient_id=f"chb0{i % 5 + 1}",
            file_path=f"/path/to/file{i}.edf",
            file_name=f"file{i}.edf",
            start_sec=i * 10,
            end_sec=i * 10 + 10,
            label=i % 2  # 交替标签
        ))
    
    sampler = BalancedSampler(seed=42)
    
    # 验证可用性
    valid, msg, stats = sampler.validate_availability(test_windows, 20)
    console.print(f"验证结果: {valid}, {msg}")
    console.print(f"统计: {stats}")
    
    # 执行采样
    result = sampler.sample(test_windows, samples_per_label=20)
    console.print(f"\n采样结果:")
    console.print(f"  Preictal: {result.preictal_count}")
    console.print(f"  Interictal: {result.interictal_count}")
    console.print(f"  涉及患者: {result.patients_sampled}")
