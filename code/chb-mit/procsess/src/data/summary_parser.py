# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""
Summary 文件解析器模块

功能：
- 解析 CHB-MIT 数据集的 summary.txt 文件
- 提取患者信息、通道配置、文件列表和癫痫发作时间
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

from ..utils.console import console, print_success, print_error, print_warning


@dataclass
class SeizureEvent:
    """
    癫痫发作事件
    
    Attributes:
        file_name: 包含发作的文件名
        start_sec: 发作开始时间（相对于文件开始，秒）
        end_sec: 发作结束时间（相对于文件开始，秒）
    """
    file_name: str
    start_sec: int
    end_sec: int
    
    @property
    def duration(self) -> int:
        """发作持续时间（秒）"""
        return self.end_sec - self.start_sec


@dataclass
class FileInfo:
    """
    EDF 文件信息
    
    Attributes:
        file_name: 文件名
        start_time: 文件开始时间（字符串格式，如 "11:42:54"）
        end_time: 文件结束时间
        duration_sec: 文件时长（秒）
        n_seizures: 文件中的发作次数
        seizures: 发作事件列表
    """
    file_name: str
    start_time: str
    end_time: str
    duration_sec: int = 0
    n_seizures: int = 0
    seizures: List[SeizureEvent] = field(default_factory=list)


@dataclass
class PatientInfo:
    """
    患者信息
    
    Attributes:
        patient_id: 患者 ID（如 "chb01"）
        sampling_rate: 采样率（Hz）
        channels: 通道列表
        files: 文件信息列表
        total_seizures: 总发作次数
        data_dir: 数据目录路径
    """
    patient_id: str
    sampling_rate: int
    channels: List[str]
    files: List[FileInfo]
    data_dir: str
    total_seizures: int = 0
    
    def get_seizure_files(self) -> List[FileInfo]:
        """获取包含发作的文件列表"""
        return [f for f in self.files if f.n_seizures > 0]
    
    def get_all_seizures(self) -> List[SeizureEvent]:
        """获取所有发作事件"""
        seizures = []
        for f in self.files:
            seizures.extend(f.seizures)
        return seizures


class SummaryParser:
    """
    Summary 文件解析器
    
    解析 CHB-MIT 数据集的 summary.txt 文件，提取：
    - 采样率
    - 通道配置
    - 文件列表和时间信息
    - 癫痫发作时间标注
    """
    
    def __init__(self, data_root: str = None):
        """
        初始化解析器
        
        Args:
            data_root: 数据集根目录
        """
        self.data_root = data_root
    
    def parse(self, patient_id: str, data_root: str = None) -> PatientInfo:
        """
        解析指定患者的 summary 文件
        
        Args:
            patient_id: 患者 ID（如 "chb01"）
            data_root: 数据集根目录（可选）
        
        Returns:
            PatientInfo 对象
        
        Raises:
            FileNotFoundError: summary 文件不存在
        """
        root = data_root or self.data_root
        if not root:
            raise ValueError("必须指定数据集根目录")
        
        patient_dir = os.path.join(root, patient_id)
        summary_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
        
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary 文件不存在: {summary_path}")
        
        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 解析采样率
        sampling_rate = self._parse_sampling_rate(content)
        
        # 解析通道列表
        channels = self._parse_channels(content)
        
        # 解析文件信息和发作时间
        files = self._parse_files(content, patient_id)
        
        # 统计总发作次数
        total_seizures = sum(f.n_seizures for f in files)
        
        patient_info = PatientInfo(
            patient_id=patient_id,
            sampling_rate=sampling_rate,
            channels=channels,
            files=files,
            data_dir=patient_dir,
            total_seizures=total_seizures
        )
        
        return patient_info
    
    def _parse_sampling_rate(self, content: str) -> int:
        """
        解析采样率
        
        Args:
            content: summary 文件内容
        
        Returns:
            采样率（Hz）
        """
        match = re.search(r'Data Sampling Rate:\s*(\d+)\s*Hz', content)
        if match:
            return int(match.group(1))
        return 256  # 默认值
    
    def _parse_channels(self, content: str) -> List[str]:
        """
        解析通道列表
        
        Args:
            content: summary 文件内容
        
        Returns:
            通道名称列表
        """
        channels = []
        
        # 匹配 "Channel N: XXXX" 格式
        pattern = r'Channel\s+(\d+):\s*(\S+)'
        matches = re.findall(pattern, content)
        
        for num, name in matches:
            channels.append(name)
        
        return channels
    
    def _parse_files(self, content: str, patient_id: str) -> List[FileInfo]:
        """
        解析文件信息和发作时间
        
        Args:
            content: summary 文件内容
            patient_id: 患者 ID
        
        Returns:
            FileInfo 列表
        """
        files = []
        
        # 按 "File Name:" 分割内容
        file_blocks = re.split(r'\n(?=File Name:)', content)
        
        for block in file_blocks:
            if not block.strip().startswith('File Name:'):
                continue
            
            file_info = self._parse_file_block(block)
            if file_info:
                files.append(file_info)
        
        return files
    
    def _parse_file_block(self, block: str) -> Optional[FileInfo]:
        """
        解析单个文件块
        
        Args:
            block: 文件信息文本块
        
        Returns:
            FileInfo 对象，解析失败返回 None
        """
        # 解析文件名
        file_match = re.search(r'File Name:\s*(\S+)', block)
        if not file_match:
            return None
        file_name = file_match.group(1)
        
        # 解析时间
        start_match = re.search(r'File Start Time:\s*(\S+)', block)
        end_match = re.search(r'File End Time:\s*(\S+)', block)
        
        start_time = start_match.group(1) if start_match else ""
        end_time = end_match.group(1) if end_match else ""
        
        # 计算时长
        duration_sec = self._calculate_duration(start_time, end_time)
        
        # 解析发作次数
        seizure_count_match = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        n_seizures = int(seizure_count_match.group(1)) if seizure_count_match else 0
        
        # 解析发作时间
        seizures = []
        if n_seizures > 0:
            start_pattern = r'Seizure\s*(?:\d+\s+)?Start Time:\s*(\d+)\s*seconds'
            end_pattern = r'Seizure\s*(?:\d+\s+)?End Time:\s*(\d+)\s*seconds'
            
            start_times = re.findall(start_pattern, block)
            end_times = re.findall(end_pattern, block)
            
            for i, (start, end) in enumerate(zip(start_times, end_times)):
                seizures.append(SeizureEvent(
                    file_name=file_name,
                    start_sec=int(start),
                    end_sec=int(end)
                ))
        
        return FileInfo(
            file_name=file_name,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration_sec,
            n_seizures=n_seizures,
            seizures=seizures
        )
    
    def _calculate_duration(self, start_time: str, end_time: str) -> int:
        """
        计算文件时长（秒）
        
        Args:
            start_time: 开始时间字符串
            end_time: 结束时间字符串
        
        Returns:
            时长（秒）
        """
        def parse_time(time_str: str) -> int:
            """解析时间字符串为秒数"""
            try:
                # 处理 "24:xx:xx" 格式（跨天）
                parts = time_str.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 3600 + m * 60 + s
            except:
                pass
            return 0
        
        start_sec = parse_time(start_time)
        end_sec = parse_time(end_time)
        
        # 处理跨天情况
        if end_sec < start_sec:
            end_sec += 24 * 3600
        
        return end_sec - start_sec


if __name__ == "__main__":
    # 测试解析器
    import argparse
    
    parser = argparse.ArgumentParser(description="Summary 文件解析器")
    parser.add_argument("--patient", "-p", required=True, help="患者 ID")
    parser.add_argument("--root", "-r", 
                       default="/mnt/chb-mit-scalp-eeg-database-1.0.0",
                       help="数据集根目录")
    args = parser.parse_args()
    
    summary_parser = SummaryParser(args.root)
    patient_info = summary_parser.parse(args.patient)
    
    console.print(f"\n[bold]患者信息: {patient_info.patient_id}[/bold]")
    console.print(f"采样率: {patient_info.sampling_rate} Hz")
    console.print(f"通道数: {len(patient_info.channels)}")
    console.print(f"文件数: {len(patient_info.files)}")
    console.print(f"总发作次数: {patient_info.total_seizures}")
    
    console.print("\n[bold]发作文件:[/bold]")
    for f in patient_info.get_seizure_files():
        console.print(f"  {f.file_name}: {f.n_seizures} 次发作")
        for s in f.seizures:
            console.print(f"    - {s.start_sec}s ~ {s.end_sec}s (持续 {s.duration}s)")
