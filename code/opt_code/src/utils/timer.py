"""
计时工具模块
============

提供函数计时功能：
- Timer: 计时器类，支持上下文管理器
- timer: 计时装饰器

使用示例
--------
>>> from src.utils.timer import timer, Timer
>>> 
>>> # 作为装饰器使用
>>> @timer
>>> def slow_function():
...     time.sleep(1)
>>> 
>>> # 作为上下文管理器使用
>>> with Timer("数据处理"):
...     process_data()
"""

import time
from functools import wraps
from typing import Callable, Optional, Any

from .logger import console, format_time


class Timer:
    """
    计时器类
    
    支持作为上下文管理器使用，自动记录并输出耗时。
    
    属性
    ----
    name : str
        计时器名称
    start_time : float
        开始时间
    end_time : float
        结束时间
    elapsed : float
        耗时（秒）
        
    使用示例
    --------
    >>> with Timer("数据处理") as t:
    ...     process_data()
    >>> print(f"耗时: {t.elapsed}秒")
    """
    
    def __init__(self, name: str = "操作", verbose: bool = True):
        """
        初始化计时器
        
        参数
        ----
        name : str
            计时器名称，用于输出
        verbose : bool
            是否输出计时信息
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        """进入上下文时开始计时"""
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        """退出上下文时停止计时"""
        self.stop()
        if self.verbose:
            self.report()
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        停止计时
        
        返回
        ----
        elapsed : float
            耗时（秒）
        """
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def report(self) -> None:
        """输出计时报告"""
        console.print(
            f"[dim]⏱[/dim] [cyan]{self.name}[/cyan] 耗时: "
            f"[bold green]{format_time(self.elapsed)}[/bold green]"
        )
    
    def reset(self) -> None:
        """重置计时器"""
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0


def timer(func: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
    """
    计时装饰器
    
    可以直接使用或带参数使用。
    
    参数
    ----
    func : Callable, optional
        被装饰的函数
    name : str, optional
        计时器名称，默认使用函数名
        
    返回
    ----
    Callable
        装饰后的函数
        
    使用示例
    --------
    >>> @timer
    >>> def process():
    ...     pass
    >>> 
    >>> @timer(name="数据处理")
    >>> def process():
    ...     pass
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            timer_name = name or fn.__name__
            with Timer(timer_name):
                result = fn(*args, **kwargs)
            return result
        return wrapper
    
    if func is not None:
        # 直接使用 @timer
        return decorator(func)
    else:
        # 带参数使用 @timer(name="xxx")
        return decorator


class TimerGroup:
    """
    计时器组，用于统计多个步骤的耗时
    
    使用示例
    --------
    >>> timers = TimerGroup()
    >>> 
    >>> with timers.timer("步骤1"):
    ...     step1()
    >>> 
    >>> with timers.timer("步骤2"):
    ...     step2()
    >>> 
    >>> timers.report()
    """
    
    def __init__(self, name: str = "任务"):
        """
        初始化计时器组
        
        参数
        ----
        name : str
            计时器组名称
        """
        self.name = name
        self.timers: dict[str, float] = {}
        self.total_start: Optional[float] = None
        self.total_elapsed: float = 0.0
    
    def start_total(self) -> None:
        """开始总计时"""
        self.total_start = time.perf_counter()
    
    def stop_total(self) -> float:
        """停止总计时"""
        if self.total_start is not None:
            self.total_elapsed = time.perf_counter() - self.total_start
        return self.total_elapsed
    
    def timer(self, name: str, verbose: bool = True) -> Timer:
        """
        获取一个命名计时器
        
        参数
        ----
        name : str
            计时器名称
        verbose : bool
            是否实时输出
            
        返回
        ----
        Timer
            计时器对象
        """
        t = Timer(name, verbose=verbose)
        
        # 保存原始的stop方法
        original_stop = t.stop
        
        def stop_and_record():
            elapsed = original_stop()
            self.timers[name] = elapsed
            return elapsed
        
        t.stop = stop_and_record
        return t
    
    def report(self) -> None:
        """输出所有计时器的报告"""
        from .logger import print_table
        
        rows = []
        for name, elapsed in self.timers.items():
            percentage = (elapsed / self.total_elapsed * 100) if self.total_elapsed > 0 else 0
            rows.append([name, format_time(elapsed), f"{percentage:.1f}%"])
        
        if self.total_elapsed > 0:
            rows.append(["[bold]总计[/bold]", format_time(self.total_elapsed), "100%"])
        
        print_table(
            f"{self.name}耗时统计",
            ["步骤", "耗时", "占比"],
            rows
        )

