import sys

class Console:
    def print(self, msg, style=None):
        print(msg)

console = Console()

def get_logger(name):
    class Logger:
        def info(self, msg):
            print(f"[INFO] {msg}")
        def warning(self, msg):
            print(f"[WARN] {msg}")
        def error(self, msg):
            print(f"[ERROR] {msg}")
    return Logger()

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step, total, desc):
    print(f"[{step}/{total}] {desc}")

def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_warning(msg):
    print(f"⚠️ {msg}")

def print_table(title, headers, rows):
    print(f"\n{title}:")
    print("  " + " | ".join(headers))
    for row in rows:
        print("  " + " | ".join(str(x) for x in row))

def print_panel(msg, title="", style=""):
    print(f"\n--- {title} ---")
    print(msg)
