"""
Standalone sanity check — confirms the DNN face detector loads and can
find a face, WITHOUT needing to launch the full Streamlit app or the
deepfake model. Run this right after download_dnn_model.py:
    python verify_dnn_detector.py path\to\some\test\image.jpg
"""
import sys
import cv2
sys.path.insert(0, "utils")
from face_utils import FaceDetector

if len(sys.argv) < 2:
    print("Usage: python verify_dnn_detector.py <path_to_test_image>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print(f"Could not read image: {img_path}")
    sys.exit(1)

d = FaceDetector()
print("DNN model loaded:", d.dnn_net is not None)
if d.dnn_net is None:
    print("!! DNN files not found/failed to load — check assets/models/dnn/ contains both files.")
    sys.exit(1)

crop = d.crop_largest_face(img)
if crop is None:
    print("No face detected in this image.")
else:
    out_path = "verify_crop_output.png"
    cv2.imwrite(out_path, crop)
    print(f"Face crop shape: {crop.shape} -> saved to {out_path}")
    print("Open that file and confirm it's actually a human face, not fur/background/etc.")
