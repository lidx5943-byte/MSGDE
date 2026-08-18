# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
窗口元数据生成
"""

import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .summary_parser import PatientInfo, FileInfo, SeizureEvent
from ..utils.console import console, print_success, print_warning, print_info, create_progress


@dataclass
class WindowSegment:
    """
    窗口片段 (用于跨文件窗口)
    """
    file_path: str
    file_name: str
    start_sec: float
    end_sec: float


@dataclass
class WindowMeta:
    """
    窗口元数据
    
    Attributes:
        patient_id: 患者 ID
        file_path: 主要文件的路径 (如果是跨文件，通常指最后一段所在的文件)
        file_name: 主要文件名
        start_sec: 在主要文件中的开始时间 (如果是跨文件，可能是负数或相对于主要文件的逻辑时间) -- 废弃，仅作兼容
        end_sec: 在主要文件中的结束时间 -- 废弃，仅作兼容
        label: 标签（0=interictal, 1=preictal）
        seizure_file: 关联的发作文件名（用于追溯）
        seizure_start: 关联发作的开始时间
        segments: 构成该窗口的文件片段列表 (新字段)
    """
    patient_id: str
    file_path: str
    file_name: str
    start_sec: float
    end_sec: float
    label: int  # 0=interictal, 1=preictal
    seizure_file: str = ""
    seizure_start: int = 0
    segments: List[WindowSegment] = None
    
    def __post_init__(self):
        if self.segments is None:
            # 向后兼容：如果没有 segments，创建一个基于 current file 的 segment
            self.segments = [
                WindowSegment(
                    file_path=self.file_path,
                    file_name=self.file_name,
                    start_sec=self.start_sec,
                    end_sec=self.end_sec
                )
            ]

    @property
    def window_id(self) -> str:
        """生成唯一标识符"""
        # 使用 segments 生成 ID 以确保唯一性
        seg_str = "|".join([f"{s.file_name}:{s.start_sec:.1f}-{s.end_sec:.1f}" for s in self.segments])
        return f"{self.patient_id}_{self.label}_{seg_str}"

    def __hash__(self):
        return hash(self.window_id)

    def __eq__(self, other):
        if not isinstance(other, WindowMeta):
            return False
        return self.window_id == other.window_id

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class WindowGenerator:
    """
    窗口元数据生成器
    
    根据配置生成 preictal 和 interictal 窗口的元数据，
    不加载实际的 EEG 数据。
    
    Preictal（发作前期）: 发作前 preictal_window 秒内的数据
    Interictal（发作间期）: 发作前 interictal_gap 秒之前的数据
    """
    
    # 标签常量
    LABEL_INTERICTAL = 0
    LABEL_PREICTAL = 1
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 提取配置参数
        self.window_size = config['preprocessing']['window_size']  # 秒
        self.overlap = config['preprocessing'].get('overlap', 0)
        self.preictal_window = config['sampling']['preictal_window']  # 发作前多少秒算 preictal
        self.interictal_gap = config['sampling']['interictal_gap']     # 发作前多少秒之前算 interictal
        self.postictal_gap = config['sampling'].get('postictal_gap', 1800)  # 发作后多少秒内不算 interictal
        
        # 计算步长
        self.step = self.window_size * (1 - self.overlap)
        if self.step <= 0:
            self.step = self.window_size
    
    def generate_for_patient(self, patient_info: PatientInfo) -> List[WindowMeta]:
        """
        为单个患者生成窗口元数据
        """
        all_windows = []
        sorted_files = self._sort_and_link_files(patient_info)
        
        # 获取所有发作事件
        seizures = patient_info.get_all_seizures()
        
        for seizure in seizures:
            preictal_windows = self._generate_preictal_windows(
                patient_info, seizure, sorted_files
            )
            all_windows.extend(preictal_windows)
        interictal_windows = self._generate_global_interictal_windows(
            patient_info, seizures, sorted_files
        )
        all_windows.extend(interictal_windows)
        
        return all_windows

    def _sort_and_link_files(self, patient_info: PatientInfo) -> List[FileInfo]:
        """返回文件列表（假设已按时间顺序）"""
        return patient_info.files

    def _get_file_predecessor(self, current_file: FileInfo, all_files: List[FileInfo]) -> Optional[FileInfo]:
        """获取前一个文件"""
        try:
            curr_idx = all_files.index(current_file)
            if curr_idx > 0:
                return all_files[curr_idx - 1]
        except ValueError:
            pass
        return None

    def _generate_preictal_windows(
        self,
        patient_info: PatientInfo,
        seizure: SeizureEvent,
        all_files: List[FileInfo]
    ) -> List[WindowMeta]:
        """生成的 Preictal 窗口，支持跨文件回溯"""
        windows = []
        
        # 找到发作所在的文件对象
        seizure_file_info = next((f for f in all_files if f.file_name == seizure.file_name), None)
        if not seizure_file_info:
            print_warning(f"未找到发作文件信息: {seizure.file_name}")
            return []

        file_path = os.path.join(patient_info.data_dir, seizure.file_name)
        
        # 从发作开始时间向后倒推生成窗口
        target_span_seconds = self.preictal_window
        current_end_offset = 0.0
        
        while current_end_offset > -target_span_seconds:
            w_end_rel = current_end_offset
            w_start_rel = current_end_offset - self.window_size
            abs_end_in_file = seizure.start_sec + w_end_rel
            abs_start_in_file = seizure.start_sec + w_start_rel
            
            segments = []
            
            # 窗口完全在当前文件内
            if abs_start_in_file >= 0:
                segments.append(WindowSegment(
                    file_path=file_path,
                    file_name=seizure.file_name,
                    start_sec=abs_start_in_file,
                    end_sec=abs_end_in_file
                ))
            
            # 窗口跨越到前一个文件
            else:
                if abs_end_in_file > 0:
                    segments.append(WindowSegment(
                        file_path=file_path,
                        file_name=seizure.file_name,
                        start_sec=0.0,
                        end_sec=abs_end_in_file
                    ))
                
                remaining_duration = -abs_start_in_file
                prev_file = self._get_file_predecessor(seizure_file_info, all_files)
                
                if prev_file and prev_file.duration_sec >= remaining_duration:
                    prev_file_path = os.path.join(patient_info.data_dir, prev_file.file_name)
                    segments.insert(0, WindowSegment(
                        file_path=prev_file_path,
                        file_name=prev_file.file_name,
                        start_sec=prev_file.duration_sec - remaining_duration,
                        end_sec=prev_file.duration_sec
                    ))
            
            # 验证窗口长度
            total_len = sum(s.end_sec - s.start_sec for s in segments)
            if abs(total_len - self.window_size) < 0.1:
                window = WindowMeta(
                    patient_id=patient_info.patient_id,
                    file_path=file_path,
                    file_name=seizure.file_name,
                    start_sec=abs_start_in_file,
                    end_sec=abs_end_in_file,
                    label=self.LABEL_PREICTAL,
                    seizure_file=seizure.file_name,
                    seizure_start=seizure.start_sec,
                    segments=segments
                )
                windows.append(window)
            
            current_end_offset -= self.step
            
        return windows

    def _generate_global_interictal_windows(
        self,
        patient_info: PatientInfo,
        seizures: List[SeizureEvent],
        all_files: List[FileInfo]
    ) -> List[WindowMeta]:
        """生成 Interictal 窗口（全局扫描，排除发作相关区域）"""
        windows = []
        file_exclusion_zones = {}  # {filename: [(start, end), ...]}
        
        for seizure in seizures:
            s_start = seizure.start_sec - self.interictal_gap
            s_end = seizure.end_sec + self.postictal_gap
            
            self._add_exclusion(file_exclusion_zones, seizure.file_name, s_start, s_end, 
                                current_file_info=next((f for f in all_files if f.file_name == seizure.file_name), None),
                                all_files=all_files)
            
        # 2. 扫描所有文件
        for file_info in all_files:
            file_path = os.path.join(patient_info.data_dir, file_info.file_name)
            if not os.path.exists(file_path):
                continue
                
            exclusions = self._merge_exclusions(file_exclusion_zones.get(file_info.file_name, []))
            duration = file_info.duration_sec or 3600
            
            current_start = 0.0
            while current_start + self.window_size <= duration:
                window_end = current_start + self.window_size
                
                # 检查是否与禁区重叠
                is_safe = True
                for ex_start, ex_end in exclusions:
                    if not (window_end <= ex_start or current_start >= ex_end):
                        is_safe = False
                        break
                
                if is_safe:
                    window = WindowMeta(
                        patient_id=patient_info.patient_id,
                        file_path=file_path,
                        file_name=file_info.file_name,
                        start_sec=current_start,
                        end_sec=window_end,
                        label=self.LABEL_INTERICTAL,
                        segments=[WindowSegment(file_path, file_info.file_name, current_start, window_end)]
                    )
                    windows.append(window)
                
                current_start += self.step
                
        return windows

    def _add_exclusion(self, zones, file_name, start_sec, end_sec, current_file_info, all_files):
        """添加禁区，处理跨文件溢出"""
        if file_name not in zones:
            zones[file_name] = []
        zones[file_name].append((start_sec, end_sec))
        
        # 处理向前溢出
        if start_sec < 0 and current_file_info:
            prev_file = self._get_file_predecessor(current_file_info, all_files)
            if prev_file:
                prev_start = prev_file.duration_sec + start_sec
                prev_end = prev_file.duration_sec
                self._add_exclusion(zones, prev_file.file_name, prev_start, prev_end, prev_file, all_files)

    def _merge_exclusions(self, exclusions):
        """合并重叠的区间"""
        if not exclusions:
            return []
        # 按开始时间排序
        sorted_ex = sorted(exclusions, key=lambda x: x[0])
        merged = []
        current_start, current_end = sorted_ex[0]
        
        for next_start, next_end in sorted_ex[1:]:
            if next_start < current_end: # 重叠或邻接
                current_end = max(current_end, next_end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged.append((current_start, current_end))
        return merged    
    def generate_all(self, patient_infos: List[PatientInfo]) -> List[WindowMeta]:
        """
        为所有患者生成窗口元数据
        
        Args:
            patient_infos: 患者信息列表
        
        Returns:
            所有窗口的 WindowMeta 列表
        """
        all_windows = []
        
        with create_progress() as progress:
            task = progress.add_task("生成窗口元数据", total=len(patient_infos))
            
            for patient_info in patient_infos:
                windows = self.generate_for_patient(patient_info)
                all_windows.extend(windows)
                progress.advance(task)
        
        # 统计
        preictal_count = sum(1 for w in all_windows if w.label == self.LABEL_PREICTAL)
        interictal_count = sum(1 for w in all_windows if w.label == self.LABEL_INTERICTAL)
        
        print_success(f"生成窗口元数据完成:")
        print_info(f"  总窗口数: {len(all_windows)}")
        print_info(f"  Preictal: {preictal_count}")
        print_info(f"  Interictal: {interictal_count}")
        
        return all_windows
    
    def save_metadata(
        self,
        windows: List[WindowMeta],
        output_path: str,
        config: Dict[str, Any] = None
    ) -> None:
        """
        保存窗口元数据到 JSON 文件
        
        Args:
            windows: 窗口元数据列表
            output_path: 输出文件路径
            config: 配置信息（可选，用于记录）
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 统计信息
        patients = list(set(w.patient_id for w in windows))
        preictal_count = sum(1 for w in windows if w.label == self.LABEL_PREICTAL)
        interictal_count = sum(1 for w in windows if w.label == self.LABEL_INTERICTAL)
        
        # 构建元数据结构
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "window_size": self.window_size,
                "overlap": self.overlap,
                "preictal_window": self.preictal_window,
                "interictal_gap": self.interictal_gap,
            },
            "class_map": {
                "0": "interictal",
                "1": "preictal"
            },
            "statistics": {
                "total_windows": len(windows),
                "preictal_windows": preictal_count,
                "interictal_windows": interictal_count,
                "patients": patients,
                "n_patients": len(patients),
            },
            "windows": [w.to_dict() for w in windows]
        }
        
        # 保存 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print_success(f"元数据已保存: {output_path}")


if __name__ == "__main__":
    # 测试窗口生成器
    from .config_loader import load_config
    from .summary_parser import SummaryParser
    
    # 加载配置
    config = load_config("config.yaml")
    
    # 解析一个患者的信息
    parser = SummaryParser(config['data']['root_path'])
    patient_info = parser.parse("chb01")
    
    # 生成窗口
    generator = WindowGenerator(config)
    windows = generator.generate_for_patient(patient_info)
    
    console.print(f"\n[bold]窗口统计:[/bold]")
    console.print(f"总窗口数: {len(windows)}")
    
    preictal = [w for w in windows if w.label == 1]
    interictal = [w for w in windows if w.label == 0]
    
    console.print(f"Preictal: {len(preictal)}")
    console.print(f"Interictal: {len(interictal)}")
