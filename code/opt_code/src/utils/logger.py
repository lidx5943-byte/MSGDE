"""
日志工具模块
============

基于Rich库提供美观的控制台输出功能：
- 彩色日志输出
- 进度条显示
- 表格展示
- 面板展示

使用示例
--------
>>> from src.utils.logger import get_logger, console, create_progress
>>> 
>>> logger = get_logger("dynamics")
>>> logger.info("开始分析...")
>>> 
>>> with create_progress() as progress:
...     task = progress.add_task("处理中...", total=100)
...     for i in range(100):
...         progress.update(task, advance=1)
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# 全局Console实例
console = Console()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    获取配置好的日志记录器
    
    参数
    ----
    name : str
        日志记录器名称
    level : int
        日志级别，默认INFO
    log_file : str, optional
        日志文件名，如果指定则同时输出到文件
    log_dir : str, optional
        日志目录，默认为当前目录下的logs
        
    返回
    ----
    logger : logging.Logger
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 确保日志传播到根日志器（这样Experiment设置的日志文件处理器才能捕获）
    logger.propagate = True
    
    # 清除现有的处理器（避免重复）
    # 但保留根日志器的处理器，只清除当前logger的处理器
    logger.handlers.clear()
    
    # 添加Rich控制台处理器（仅用于控制台输出）
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)
    
    # 如果指定了日志文件，添加额外的文件处理器
    # 注意：这不会阻止日志传播到根日志器
    if log_file or log_dir:
        if log_dir is None:
            log_dir = "logs"
        
        os.makedirs(log_dir, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_file)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件: {log_path}")
    
    return logger


def create_progress(
    description: str = "处理中",
    transient: bool = False,
) -> Progress:
    """
    创建Rich进度条
    
    参数
    ----
    description : str
        进度条描述
    transient : bool
        完成后是否自动消失
        
    返回
    ----
    progress : Progress
        Progress对象，可用作上下文管理器
        
    使用示例
    --------
    >>> with create_progress() as progress:
    ...     task = progress.add_task("处理数据", total=100)
    ...     for i in range(100):
    ...         # 处理逻辑
    ...         progress.update(task, advance=1)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def print_table(
    title: str,
    columns: List[str],
    rows: List[List[Any]],
    caption: Optional[str] = None,
) -> None:
    """
    打印格式化表格
    
    参数
    ----
    title : str
        表格标题
    columns : List[str]
        列名列表
    rows : List[List[Any]]
        行数据列表
    caption : str, optional
        表格说明
        
    使用示例
    --------
    >>> print_table(
    ...     "分析结果",
    ...     ["指标", "值"],
    ...     [["均值", "0.5"], ["方差", "0.1"]]
    ... )
    """
    table = Table(
        title=title,
        caption=caption,
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold magenta",
    )
    
    for col in columns:
        table.add_column(col, justify="center")
    
    for row in rows:
        table.add_row(*[str(item) for item in row])
    
    console.print(table)


def print_panel(
    content: str,
    title: str = "",
    subtitle: str = "",
    style: str = "green",
) -> None:
    """
    打印面板信息
    
    参数
    ----
    content : str
        面板内容
    title : str
        面板标题
    subtitle : str
        面板副标题
    style : str
        面板样式颜色
        
    使用示例
    --------
    >>> print_panel("分析完成！", title="成功", style="green")
    """
    panel = Panel(
        Text(content, justify="center"),
        title=title,
        subtitle=subtitle,
        border_style=style,
        box=box.DOUBLE,
    )
    console.print(panel)


def print_header(text: str, style: str = "bold magenta") -> None:
    """
    打印标题头
    
    参数
    ----
    text : str
        标题文本
    style : str
        样式
    """
    console.print()
    console.rule(f"[{style}]{text}[/{style}]")
    console.print()


def print_step(step: int, total: int, description: str) -> None:
    """
    打印步骤信息
    
    参数
    ----
    step : int
        当前步骤
    total : int
        总步骤数
    description : str
        步骤描述
    """
    console.print(f"[bold cyan]步骤 [{step}/{total}][/bold cyan] {description}")


def print_success(message: str) -> None:
    """打印成功消息"""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str) -> None:
    """打印警告消息"""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_error(message: str) -> None:
    """打印错误消息"""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_info(message: str) -> None:
    """打印信息消息"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_dict(data: Dict[str, Any], title: str = "配置信息") -> None:
    """
    打印字典信息为表格形式
    
    参数
    ----
    data : Dict[str, Any]
        要显示的字典
    title : str
        表格标题
    """
    rows = [[key, str(value)] for key, value in data.items()]
    print_table(title, ["参数", "值"], rows)


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    参数
    ----
    seconds : float
        秒数
        
    返回
    ----
    str
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

