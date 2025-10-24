import os
import signal
import subprocess
import sys
from pathlib import Path

import cv2

# Set timeout for the script (900 seconds)
signal.alarm(900)

runner_os = os.environ.get("RUNNER_OS", "")

# Create dirs
os.makedirs("test_images/tiles", exist_ok=True)
os.makedirs("test_images/upscaled", exist_ok=True)

# Create test image
subprocess.run([sys.executable, "create_test_image.py"], check=False)

# Preprocess with C++ version
if runner_os == "Windows":
    subprocess.run(["./build/Release/preprocess.exe", "test_images", "test_images/tiles"], check=False)
else:
    subprocess.run(["./build/preprocess", "test_images", "test_images/tiles"], check=False)

# Also test C version
if runner_os == "Windows":
    subprocess.run(["./build/Release/preprocess_c.exe", "test_images", "test_images/tiles_c"], check=False)
else:
    subprocess.run(["./build/preprocess_c", "test_images", "test_images/tiles_c"], check=False)

# Verify C version produced tiles
if os.path.exists("test_images/tiles_c"):
    c_tiles = len([f for f in Path("test_images/tiles_c").iterdir() if f.suffix == ".jpg"])
    if c_tiles != 16:
        sys.exit(1)

print("Preprocess done")

# Tiles are already in test_images/tiles/

# Upscale tiles using the hybrid backend (Metal on macOS, CUDA on Linux)
for i in range(16):
    input_tile = f"test_images/tiles/test_tile_{i}.jpg"
    output_tile = f"test_images/upscaled/tile_{i}.jpg"
    if os.path.exists(input_tile):
        if runner_os == "Windows":
            subprocess.run(["./build/Release/upscaler.exe", input_tile, output_tile], check=False)
        else:
            subprocess.run(["./build/upscaler", input_tile, output_tile], check=False)

        # Verify upscale
        if os.path.exists(output_tile):
            img = cv2.imread(output_tile)
            if img is not None and img.shape[1] == 128 and img.shape[0] == 128:  # 64x64 * 2
                continue
        sys.exit(1)
print("Upscale done")

# Stitch
if runner_os == "Windows":
    subprocess.run([sys.executable, "scripts/stitch.py", "test_images/upscaled", "test_images/final_output.jpg"], check=False)
else:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--source=scripts",
            "scripts/stitch.py",
            "test_images/upscaled",
            "test_images/final_output.jpg",
        ],
        check=False,
    )

print("Stitch done")

# Verify
if os.path.exists("test_images/final_output.jpg"):
    img = cv2.imread("test_images/final_output.jpg")
    if img is not None and img.shape[1] == 512 and img.shape[0] == 512:
        print("E2E test passed")
    else:
        sys.exit(1)
else:
    sys.exit(1)
