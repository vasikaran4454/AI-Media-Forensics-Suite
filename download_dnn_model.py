"""
One-time setup: downloads the DNN face detector model files.
Run this once from the project root (venv activated):
    python download_dnn_model.py
"""
import os
import urllib.request

DNN_DIR = os.path.join("assets", "models", "dnn")
os.makedirs(DNN_DIR, exist_ok=True)

FILES = {
    "deploy.prototxt":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel":
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}

for fname, url in FILES.items():
    dest = os.path.join(DNN_DIR, fname)
    print(f"Downloading {fname} ...")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  -> saved to {dest} ({size:,} bytes)")
    if size < 1000:
        print(f"  !! WARNING: {fname} looks too small — download may have failed. Check the file content.")

print("\nDone. Run: python verify_dnn_detector.py   to confirm it works before running the app.")
