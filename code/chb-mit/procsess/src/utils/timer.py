import time
from contextlib import contextmanager

class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

class TimerGroup:
    def __init__(self, name):
        self.name = name
        self.timers = {}
        self.total_start = None
        self.total_elapsed = 0

    def start_total(self):
        self.total_start = time.time()

    def stop_total(self):
        self.total_elapsed = time.time() - self.total_start

    @contextmanager
    def timer(self, name):
        t = Timer(name)
        with t:
            yield
        self.timers[name] = t.elapsed

    def report(self):
        print(f"\⏱️ {self.name} 耗时统计:")
        for name, elapsed in self.timers.items():
            print(f"  {name}: {elapsed:.3f}s")
        print(f"  总计: {self.total_elapsed:.3f}s")