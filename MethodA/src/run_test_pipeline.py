"""
run_test_pipeline.py

Runs the full IF-category test pipeline in order:
  1. classify_if_anomalies.py  — classify IF anomalies into categories
  2. plot_if_categories.py     — generate per-category figures

Run from the project root:
  python test/src/run_test_pipeline.py
"""

import subprocess
import sys
import os


STEPS = [
    ("classify_if_anomalies.py", "Classifying IF anomalies into categories"),
    ("plot_if_categories.py",    "Generating per-category figures"),
]


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, "..", ".."))

    print(f"Project root: {project_root}")
    print(f"Script dir  : {src_dir}\n")

    for script, desc in STEPS:
        script_path = os.path.join(src_dir, script)
        cmd = [sys.executable, script_path]

        print("=" * 60)
        print(f"Step: {desc}")
        print(f"  -> {' '.join(cmd)}")
        print("=" * 60)

        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode != 0:
            print(f"\nERROR: {script} failed (exit code {result.returncode})")
            sys.exit(1)
        print()

    print("=" * 60)
    print("Pipeline complete.  Outputs in test/output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
