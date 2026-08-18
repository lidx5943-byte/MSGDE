"""
数据IO工具模块
==============

提供数据输入输出功能：
- 文件保存与加载
- 目录管理
- 实验管理
- 输出路径生成

使用示例
--------
>>> from src.utils.io import save_numpy, load_numpy, Experiment
>>> 
>>> # 创建实验
>>> exp = Experiment("dynamics_analysis", config_path="config/config.yaml")
>>> exp.log_command(sys.argv)
>>> 
>>> # 保存数据
>>> save_numpy(data, exp.data_dir / "features.npy")
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Any, Dict, List

import numpy as np
import yaml

from .logger import console, print_success, print_error, print_info, print_panel, get_logger


class Experiment:
    """
    实验管理类
    
    管理单次实验的目录结构、配置文件和运行记录。
    
    目录结构
    --------
    output/
    └── {experiment_name}_{timestamp}/
        ├── config/
        │   ├── config.yaml          # 配置文件副本
        │   └── command.txt          # 运行命令
        ├── data/
        │   ├── input/               # 输入数据链接/副本
        │   └── output/              # 输出数据
        ├── figures/
        ├── logs/
        └── reports/
            └── experiment_info.json  # 实验信息
    
    使用示例
    --------
    >>> exp = Experiment("dynamics", config_path="config/config.yaml")
    >>> exp.log_command(sys.argv)
    >>> exp.save_input_info({"data": "path/to/data.npy"})
    >>> 
    >>> # 在脚本中使用
    >>> save_numpy(result, exp.data_dir / "output" / "result.npy")
    """
    
    def __init__(
        self,
        name: str,
        base_dir: Union[str, Path] = "output",
        config_path: Union[str, Path] = None,
        timestamp: str = None,
        description: str = "",
        create_all_dirs: bool = True,
    ):
        """
        初始化实验
        
        参数
        ----
        name : str
            实验名称
        base_dir : str or Path
            基础输出目录
        config_path : str or Path, optional
            配置文件路径，会复制到实验目录
        timestamp : str, optional
            时间戳，默认自动生成
        description : str
            实验描述
        create_all_dirs : bool
            是否创建所有目录（包括data、figures等），False时只创建必要目录
        """
        self.name = name
        self.description = description
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.create_all_dirs = create_all_dirs
        
        # 创建实验目录
        self.root_dir = Path(base_dir) / f"{name}_{self.timestamp}"
        self._create_directories()
        
        # 复制配置文件
        self.config_path = None
        if config_path:
            self._copy_config(config_path)
        
        # 初始化实验信息
        self.info = {
            "name": name,
            "description": description,
            "timestamp": self.timestamp,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "inputs": {},
            "outputs": {},
        }
        self._save_info()
        
        # 设置日志记录
        self._setup_logging()
        
        # 打印实验信息
        print_panel(
            f"实验名称: {name}\n"
            f"时间戳: {self.timestamp}\n"
            f"目录: {self.root_dir}",
            title="🧪 实验初始化",
            style="blue"
        )
    
    def _create_directories(self):
        """创建目录结构"""
        # 总是创建必要的目录
        self.config_dir = self.root_dir / "config"
        self.logs_dir = self.root_dir / "logs"
        self.reports_dir = self.root_dir / "reports"
        
        required_dirs = [self.config_dir, self.logs_dir, self.reports_dir]
        
        # 根据create_all_dirs决定是否创建其他目录
        if self.create_all_dirs:
            self.data_dir = self.root_dir / "data"
            self.input_dir = self.data_dir / "input"
            self.output_dir = self.data_dir / "output"
            self.figures_dir = self.root_dir / "figures"
            required_dirs.extend([self.input_dir, self.output_dir, self.figures_dir])
        else:
            # 设置为None，使用时再创建
            self.data_dir = None
            self.input_dir = None
            self.output_dir = None
            self.figures_dir = None
        
        # 创建目录
        for d in required_dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志记录到实验目录的logs文件夹"""
        # 创建实验日志文件路径
        log_file = f"experiment_{self.timestamp}.log"
        log_path = self.logs_dir / log_file
        
        # 获取根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 检查是否已经添加了相同路径的文件处理器（避免重复添加）
        log_path_str = str(log_path.absolute())
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and 
            h.baseFilename == log_path_str
            for h in root_logger.handlers
        )
        
        if not has_file_handler:
            # 添加文件处理器
            file_handler = logging.FileHandler(log_path, encoding="utf-8", mode='w')
            file_handler.setLevel(logging.DEBUG)
            
            # 设置格式
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            
            root_logger.addHandler(file_handler)
            
            # 记录日志文件路径到实验信息
            self.info["log_file"] = str(log_path)
            self._save_info()
            
            # 记录实验开始信息
            logger = logging.getLogger(__name__)
            logger.info("=" * 80)
            logger.info(f"Experiment: {self.name}")
            logger.info(f"Timestamp: {self.timestamp}")
            logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Log File: {log_path}")
            logger.info("=" * 80)
        
        # 确保所有模块的日志记录器都使用根日志器的处理器
        # 通过设置 propagate=True，子日志器会将日志传播到根日志器
        module_loggers = [
            "src",
            "src.preprocessing",
            "src.similarity",
            "src.laplacian",
            "src.dynamics",
            "src.visualization",
            "src.ml",
            "src.utils",
        ]
        
        for module_name in module_loggers:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)
            module_logger.propagate = True  # 确保日志传播到根日志器
    
    def _copy_config(self, config_path: Union[str, Path]):
        """复制配置文件到实验目录"""
        config_path = Path(config_path)
        if config_path.exists():
            dest = self.config_dir / config_path.name
            shutil.copy2(config_path, dest)
            self.config_path = dest
            
            # 同时加载配置内容到info
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.info["config"] = yaml.safe_load(f)
            except:
                pass
            
            print_success(f"Config saved: {dest}")
    
    def setup_logging(self):
        """
        Setup file logging for the experiment
        
        Creates a log file in the logs directory that captures console output.
        """
        import logging
        
        # Create log file
        log_file = self.logs_dir / f"experiment_{self.timestamp}.log"
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        self.log_file = log_file
        self.info["log_file"] = str(log_file)
        self._save_info()
        
        print_success(f"Logging to: {log_file}")
        
        return log_file
    
    def save_full_config_report(self):
        """
        Save a comprehensive configuration report in human-readable format
        """
        if not hasattr(self, 'info') or 'config' not in self.info:
            return
        
        config = self.info.get('config', {})
        report_path = self.reports_dir / "full_config_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FULL CONFIGURATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Experiment: {self.name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            def write_section(f, title, data, indent=0):
                prefix = "  " * indent
                f.write(f"{prefix}{'-' * (80 - indent*2)}\n")
                f.write(f"{prefix}{title}\n")
                f.write(f"{prefix}{'-' * (80 - indent*2)}\n")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            f.write(f"{prefix}  {key}:\n")
                            for k, v in value.items():
                                f.write(f"{prefix}    {k}: {v}\n")
                        else:
                            f.write(f"{prefix}  {key}: {value}\n")
                f.write("\n")
            
            # Write each config section
            if 'preprocessing' in config:
                write_section(f, "PREPROCESSING", config['preprocessing'])
            
            if 'similarity' in config:
                write_section(f, "SIMILARITY", config['similarity'])
            
            if 'laplacian' in config:
                write_section(f, "LAPLACIAN", config['laplacian'])
            
            if 'dynamics' in config:
                write_section(f, "DYNAMICS", config['dynamics'])
            
            if 'visualization' in config:
                write_section(f, "VISUALIZATION", config['visualization'])
            
            if 'output' in config:
                write_section(f, "OUTPUT", config['output'])
            
            if 'parallel' in config:
                write_section(f, "PARALLEL", config['parallel'])
            
            f.write("=" * 80 + "\n")
        
        print_success(f"Full config report saved: {report_path}")
    
    def _save_info(self):
        """保存实验信息"""
        info_path = self.reports_dir / "experiment_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.info, f, indent=2, ensure_ascii=False)
    
    def log_command(self, argv: List[str] = None):
        """
        记录运行命令
        
        参数
        ----
        argv : List[str], optional
            命令行参数，默认使用sys.argv
        """
        if argv is None:
            argv = sys.argv
        
        command = " ".join(argv)
        self.info["command"] = command
        
        # 保存到文件
        command_path = self.config_dir / "command.txt"
        with open(command_path, 'w', encoding='utf-8') as f:
            f.write(f"# 运行时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 工作目录: {os.getcwd()}\n\n")
            f.write(command + "\n")
        
        # 同时保存为shell脚本
        script_path = self.config_dir / "run.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# 实验: {self.name}\n")
            f.write(f"# 时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write(command + "\n")
        
        self._save_info()
        print_success(f"命令已保存: {command_path}")
    
    def log_input(self, name: str, path: Union[str, Path], copy: bool = False):
        """
        记录输入数据
        
        参数
        ----
        name : str
            输入名称
        path : str or Path
            输入文件路径
        copy : bool
            是否复制文件到实验目录
        """
        path = Path(path)
        self.info["inputs"][name] = str(path.absolute())
        
        if copy and path.exists():
            # 如果input_dir不存在，创建它
            if self.input_dir is None:
                self.input_dir = self.root_dir / "data" / "input"
                self.input_dir.mkdir(parents=True, exist_ok=True)
            dest = self.input_dir / path.name
            shutil.copy2(path, dest)
            self.info["inputs"][f"{name}_copy"] = str(dest)
        
        self._save_info()
    
    def log_output(self, name: str, path: Union[str, Path]):
        """记录输出数据"""
        self.info["outputs"][name] = str(Path(path).absolute())
        self._save_info()
    
    def log_metric(self, name: str, value: Any):
        """记录指标"""
        if "metrics" not in self.info:
            self.info["metrics"] = {}
        self.info["metrics"][name] = value
        self._save_info()
    
    def finish(self, status: str = "completed"):
        """
        Finish the experiment
        
        Parameters
        ----------
        status : str
            Experiment status: completed, failed, interrupted
        """
        self.info["status"] = status
        self.info["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.info["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()
        self._save_info()
        
        # Save full config report
        self.save_full_config_report()
        
        # Save console log summary
        self._save_log_summary()
        
        print_panel(
            f"Status: {status}\n"
            f"Duration: {self.info['duration_seconds']:.1f} seconds\n"
            f"Directory: {self.root_dir}",
            title="Experiment Finished",
            style="green" if status == "completed" else "red"
        )
    
    def _save_log_summary(self):
        """Save experiment log summary"""
        log_path = self.logs_dir / "experiment_summary.log"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT SUMMARY LOG\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Experiment Name: {self.name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Status: {self.info.get('status', 'unknown')}\n")
            f.write(f"Start Time: {self.info.get('start_time', 'N/A')}\n")
            f.write(f"End Time: {self.info.get('end_time', 'N/A')}\n")
            f.write(f"Duration: {self.info.get('duration_seconds', 0):.1f} seconds\n\n")
            
            if self.info.get('command'):
                f.write("-" * 60 + "\n")
                f.write("COMMAND\n")
                f.write("-" * 60 + "\n")
                f.write(f"{self.info['command']}\n\n")
            
            if self.info.get('inputs'):
                f.write("-" * 60 + "\n")
                f.write("INPUTS\n")
                f.write("-" * 60 + "\n")
                for name, path in self.info['inputs'].items():
                    f.write(f"  {name}: {path}\n")
                f.write("\n")
            
            if self.info.get('outputs'):
                f.write("-" * 60 + "\n")
                f.write("OUTPUTS\n")
                f.write("-" * 60 + "\n")
                for name, path in self.info['outputs'].items():
                    f.write(f"  {name}: {path}\n")
                f.write("\n")
            
            if self.info.get('metrics'):
                f.write("-" * 60 + "\n")
                f.write("METRICS\n")
                f.write("-" * 60 + "\n")
                for name, value in self.info['metrics'].items():
                    f.write(f"  {name}: {value}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.finish("failed")
        else:
            self.finish("completed")
        return False


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如不存在则创建
    
    参数
    ----
    path : str or Path
        目录路径
        
    返回
    ----
    Path
        目录路径对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(
    experiment_name: str = "experiment",
    base_dir: Union[str, Path] = "output",
    timestamp: Optional[str] = None,
    create_subdirs: bool = True,
) -> Path:
    """
    生成带时间戳的输出目录
    
    参数
    ----
    experiment_name : str
        实验名称
    base_dir : str or Path
        基础输出目录
    timestamp : str, optional
        时间戳，默认自动生成
    create_subdirs : bool
        是否创建子目录（logs, data, figures, reports）
        
    返回
    ----
    Path
        输出目录路径
        
    目录结构
    --------
    output/
    └── {experiment_name}_{timestamp}/
        ├── logs/
        ├── data/
        ├── figures/
        └── reports/
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    ensure_dir(output_dir)
    
    if create_subdirs:
        subdirs = ["logs", "data", "figures", "reports"]
        for subdir in subdirs:
            ensure_dir(output_dir / subdir)
    
    print_info(f"输出目录: {output_dir}")
    
    return output_dir


def save_numpy(
    data: np.ndarray,
    path: Union[str, Path],
    allow_pickle: bool = False,
    compress: bool = False,
) -> Path:
    """
    保存NumPy数组
    
    参数
    ----
    data : np.ndarray
        要保存的数据
    path : str or Path
        保存路径
    allow_pickle : bool
        是否允许pickle序列化
    compress : bool
        是否压缩保存（使用.npz格式）
        
    返回
    ----
    Path
        保存的文件路径
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    try:
        if compress:
            # 压缩保存
            if not path.suffix == ".npz":
                path = path.with_suffix(".npz")
            np.savez_compressed(path, data=data)
        else:
            # 普通保存
            if not path.suffix == ".npy":
                path = path.with_suffix(".npy")
            np.save(path, data, allow_pickle=allow_pickle)
        
        # 获取文件大小
        size_mb = path.stat().st_size / (1024 * 1024)
        print_success(f"已保存: {path} ({size_mb:.2f} MB)")
        
        return path
        
    except Exception as e:
        print_error(f"保存失败: {path} - {e}")
        raise


def load_numpy(
    path: Union[str, Path],
    allow_pickle: bool = False,
) -> np.ndarray:
    """
    加载NumPy数组
    
    参数
    ----
    path : str or Path
        文件路径
    allow_pickle : bool
        是否允许pickle反序列化
        
    返回
    ----
    np.ndarray
        加载的数据
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=allow_pickle) as data:
                # 如果是单个数组，直接返回
                if "data" in data:
                    return data["data"]
                else:
                    # 返回第一个数组
                    return data[list(data.keys())[0]]
        else:
            data = np.load(path, allow_pickle=allow_pickle)
        
        # 获取文件大小
        size_mb = path.stat().st_size / (1024 * 1024)
        print_success(f"已加载: {path} ({size_mb:.2f} MB)")
        
        return data
        
    except Exception as e:
        print_error(f"加载失败: {path} - {e}")
        raise


def save_dict(
    data: Dict[str, np.ndarray],
    path: Union[str, Path],
    compress: bool = True,
) -> Path:
    """
    保存多个数组为一个npz文件
    
    参数
    ----
    data : Dict[str, np.ndarray]
        数据字典，键为数组名
    path : str or Path
        保存路径
    compress : bool
        是否压缩
        
    返回
    ----
    Path
        保存的文件路径
    """
    path = Path(path)
    if not path.suffix == ".npz":
        path = path.with_suffix(".npz")
    
    ensure_dir(path.parent)
    
    try:
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
        
        size_mb = path.stat().st_size / (1024 * 1024)
        print_success(f"已保存: {path} ({size_mb:.2f} MB)")
        
        return path
        
    except Exception as e:
        print_error(f"保存失败: {path} - {e}")
        raise


def load_dict(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    从npz文件加载多个数组
    
    参数
    ----
    path : str or Path
        文件路径
        
    返回
    ----
    Dict[str, np.ndarray]
        数据字典
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    try:
        with np.load(path) as npz_file:
            data = {key: npz_file[key] for key in npz_file.files}
        
        size_mb = path.stat().st_size / (1024 * 1024)
        print_success(f"已加载: {path} ({size_mb:.2f} MB)")
        
        return data
        
    except Exception as e:
        print_error(f"加载失败: {path} - {e}")
        raise


def list_files(
    directory: Union[str, Path],
    pattern: str = "*.npy",
    recursive: bool = False,
) -> list:
    """
    列出目录中的文件
    
    参数
    ----
    directory : str or Path
        目录路径
    pattern : str
        文件匹配模式
    recursive : bool
        是否递归搜索
        
    返回
    ----
    list
        文件路径列表
    """
    directory = Path(directory)
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    return sorted(files)


def get_file_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件信息
    
    参数
    ----
    path : str or Path
        文件路径
        
    返回
    ----
    Dict[str, Any]
        文件信息字典
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    stat = path.stat()
    
    info = {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 如果是numpy文件，获取数组形状
    if path.suffix in [".npy", ".npz"]:
        try:
            data = np.load(path, allow_pickle=True)
            if hasattr(data, "shape"):
                info["shape"] = data.shape
                info["dtype"] = str(data.dtype)
            elif hasattr(data, "files"):
                info["arrays"] = data.files
        except:
            pass
    
    return info

