# -*- coding: utf-8 -*-
# Author: 王梓涵 <wangzh011031@163.com>
"""Rich 控制台：标题、步骤、消息、进度条。"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

console = Console()


def print_header(title: str, subtitle: str = None) -> None:
    """标题面板。"""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    
    panel = Panel(
        content,
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(panel)


def print_step(step_num: int, total_steps: int, description: str) -> None:
    """步骤进度。"""
    console.print(f"\n[bold blue]━━━ 步骤 {step_num}/{total_steps}: {description} ━━━[/bold blue]")


def print_success(message: str) -> None:
    """成功。"""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """错误。"""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """警告。"""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """信息。"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def create_progress() -> Progress:
    """进度条。"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    )


def print_statistics(stats: dict, title: str = "统计信息") -> None:
    """统计表格。"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    for key, value in stats.items():
        table.add_row(str(key), str(value))
    
    console.print(table)
