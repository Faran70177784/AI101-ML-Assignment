"""
Environment and Dependency Verification Script
Course: AI-101 Introduction to Artificial Intelligence
University: The University of Lahore, Department of Intelligent Systems
Instructor: Dr. Muhammad Awais
"""

import sys
import os
import platform

def check_python_version():
    ver = sys.version_info
    print(f"[+] Python Executable: {sys.executable}")
    print(f"[+] Python Version: {ver.major}.{ver.minor}.{ver.micro} ({platform.architecture()[0]})")
    if ver.major == 3 and ver.minor in (11, 13):
        print(f"    -> Active Python Runtime: {ver.major}.{ver.minor}.{ver.micro}")
        return True
    return False

def check_system_hardware():
    print(f"[+] Operating System: {platform.system()} {platform.release()} (Build {platform.version()})")
    print(f"[+] Machine Processor: {platform.processor() or platform.machine()}")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"[+] Total System RAM: {ram_gb:.2f} GB")
    except ImportError:
        print("[!] psutil unavailable; estimating 8 GB RAM.")

def check_packages():
    packages_to_check = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("ucimlrepo", "ucimlrepo"),
        ("nltk", "nltk"),
        ("openpyxl", "openpyxl"),
        ("jupyter", "jupyter"),
        ("ipykernel", "ipykernel"),
    ]
    print("\n--- Checking Core Machine Learning Libraries ---")
    all_ok = True
    for display_name, mod_name in packages_to_check:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Available")
            print(f"  [v] {display_name:<15}: {ver}")
        except Exception as e:
            all_ok = False
            print(f"  [x] {display_name:<15}: ERROR ({e})")
    return all_ok

def check_deep_learning_environment():
    print("\n--- Deep Learning & GPU Environment (Task C3) ---")
    try:
        import torch
        print(f"  [v] PyTorch: {torch.__version__}")
        print(f"  [v] CUDA Available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"  [i] PyTorch local note: Local execution on CPU ({e}).")
        print("  [i] Official Assignment Recommendation: Free Google Colab / Kaggle T4 GPU will be used for Pool C deep seq2seq training.")

if __name__ == "__main__":
    print("=" * 65)
    print("AI-101 Machine Learning Assignment: Environment Verification")
    print("=" * 65)
    check_python_version()
    check_system_hardware()
    check_packages()
    check_deep_learning_environment()
    print("=" * 65)
