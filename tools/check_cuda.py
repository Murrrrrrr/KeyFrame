import subprocess
import sys
import torch

print("=" * 50)
print("CUDA Environment Check")
print("=" * 50)

# -----------------------------
# 1. Python version
# -----------------------------
print(f"Python version: {sys.version}\n")

# -----------------------------
# 2. NVIDIA GPU detection (nvidia-smi)
# -----------------------------
print("[1] Checking NVIDIA GPU (nvidia-smi)...")

try:
    result = subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print("✅ NVIDIA GPU detected")
        print(result.stdout.split("\n")[0])
    else:
        print("❌ nvidia-smi exists but failed")
except FileNotFoundError:
    print("❌ nvidia-smi not found (No NVIDIA driver?)")

print()

# -----------------------------
# 3. PyTorch CUDA check
# -----------------------------
print("[2] Checking PyTorch CUDA...")

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version (torch): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"Memory: "
                f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            )
    else:
        print("⚠️ CUDA not available in PyTorch")

except ImportError:
    print("❌ PyTorch not installed")

print()

# -----------------------------
# 4. nvcc compiler check
# -----------------------------
print("[3] Checking CUDA Toolkit (nvcc)...")

try:
    result = subprocess.run(
        ["nvcc", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        print("✅ CUDA Toolkit installed")
        print(result.stdout.split("\n")[-3])
    else:
        print("❌ nvcc exists but failed")
except FileNotFoundError:
    print("❌ nvcc not found (CUDA Toolkit missing)")

print("\nCheck finished.")
print("=" * 50)

print(torch.__version__)          # 看看版本号后面有没有带 +cu121 或 +cu118 字样
print(torch.cuda.is_available())  # 如果输出 True，恭喜你，大功告成！
print(torch.cuda.get_device_name(0)) # 这会打印出你的显卡型号，比如 "Tesla T4" 或 "RTX 4090"
