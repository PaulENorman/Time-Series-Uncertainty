"""Reproduce the current article figures and HTML in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "generate_article_figures.py",
    "render_docs.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===", flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=False)
        if proc.returncode != 0:
            print(f"Failed: {script} (exit {proc.returncode})", flush=True)
            return proc.returncode
    print("\nAll scripts completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
