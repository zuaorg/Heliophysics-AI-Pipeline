#!/usr/bin/env python
"""
Script to run all training or testing commands in one go.

Usage:
    python run_all.py train 24   # Runs all training scripts with variable = 24
    python run_all.py test 24    # Runs all testing scripts with variable = 24

Note:
    The numeric variable (e.g., 24) can be changed as needed.
"""

import subprocess
import sys

# Update these paths with your actual directories
MLAAS_DIR = "flare-prediction"
CME_DIR = "CME-prediction"
SEP_DIR = "SEP-prediction"


def run_train(var):
    print(f"Running training scripts with variable: {var}")

    # Run mlaas_train.py
    subprocess.run(["python", "mlaas_train.py", "-m", "new"], cwd=MLAAS_DIR, check=True)

    # Run CMEpredict.py in training mode (last argument = "1")
    subprocess.run(["python", "CMEpredict.py", "gru", str(var), "1"], cwd=CME_DIR, check=True)

    # Run SEP_train.py
    subprocess.run(["python", "SEP_train.py", "F_S", str(var)], cwd=SEP_DIR, check=True)


def run_test(var):
    print(f"Running testing scripts with variable: {var}")

    # Run mlaas_test.py
    subprocess.run(["python", "mlaas_test.py", "-m", "new"], cwd=MLAAS_DIR, check=True)

    # Run CMEpredict.py in testing mode (last argument = "0")
    subprocess.run(["python", "CMEpredict.py", "ens", str(var), "0"], cwd=CME_DIR, check=True)

    # Run SEP_test.py
    subprocess.run(["python", "SEP_test.py", "F_S", str(var)], cwd=SEP_DIR, check=True)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_pipeline.py [train|test] <variable>")
        sys.exit(1)

    mode = sys.argv[1].lower()
    var = sys.argv[2]

    if mode == "train":
        run_train(var)
    elif mode == "test":
        run_test(var)
    else:
        print("Error: Mode must be either 'train' or 'test'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
