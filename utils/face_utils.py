"""
Face detection utility.
Primary: OpenCV's DNN face detector (ResNet-10 SSD, Caffe) — much more
robust than Haar cascade to angle/pose/lighting, which matters for
composite/AI-generated images with non-frontal faces. Falls back to Haar
cascade automatically if the DNN model files aren't present, so this
still works out of the box with zero extra downloads — just less
reliably on tricky poses (see fallback docstring for the confirmed
failure case that motivated adding the DNN path).

To enable the DNN detector (recommended), download these two files into
assets/models/dnn/ (paths relative to the project root):

    deploy.prototxt
      https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    res10_300x300_ssd_iter_140000.caffemodel
      https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

On Windows (from the project root, venv activated):
    curl -o assets\\models\\dnn\\deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    curl -L -o assets\\models\\dnn\\res10_300x300_ssd_iter_140000.caffemodel https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
"""

import cv2
import numpy as np
import os

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

DNN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models", "dnn")
DNN_PROTOTXT = os.path.join(DNN_DIR, "deploy.prototxt")
DNN_WEIGHTS = os.path.join(DNN_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(CASCADE_PATH)

        self.dnn_net = None
        if os.path.exists(DNN_PROTOTXT) and os.path.exists(DNN_WEIGHTS):
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_WEIGHTS)
            except Exception:
                self.dnn_net = None  # corrupt/partial download — fall back silently

    # ------------------------------------------------------------
    # DNN detector (preferred) — robust to angle/pose
    # ------------------------------------------------------------
    def _detect_faces_dnn(self, image_bgr, conf_threshold=0.5):
        """
        Returns list of (x, y, w, h, confidence) using the ResNet-10 SSD
        face detector. confidence is a true 0-1 detection probability
        (unlike Haar's unbounded levelWeights), so it's directly
        comparable and meaningful for ranking multiple faces in a busy
        image.
        """
        H, W = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < conf_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                results.append((x1, y1, w, h, conf))
        return results

    # ------------------------------------------------------------
    # Haar cascade (fallback) — works with zero extra downloads, but
    # confirmed unreliable on angled/non-frontal faces in busy images.
    #
    # CONFIRMED FAILURE CASE (kept here for reference): on a composite
    # image of a person looking down/sideways at a cat, the Haar
    # cascade's OWN confidence score (levelWeights) rated a false
    # positive on the cat's back fur at 6.58, while the person's real,
    # angled face scored only 1.27 — LOWER than every other false
    # positive in the image (kittens, fur). Ranking by Haar confidence
    # instead of box area (the previous fix) helps when the real face
    # IS a strong frontal match; it cannot help when Haar's own
    # detector genuinely rates the real face as a weak match. That's
    # a detector-capability limit, not a ranking bug — hence the DNN
    # path above, which does not have this weakness.
    # ------------------------------------------------------------
    def _detect_faces_haar(self, image_bgr, min_size=(60, 60)):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes, reject_levels, level_weights = self.cascade.detectMultiScale3(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=min_size,
            outputRejectLevels=True
        )
        results = []
        for (x, y, w, h), weight in zip(boxes, level_weights):
            results.append((x, y, w, h, float(weight)))
        return results

    def detect_faces(self, image_bgr, min_size=(60, 60)):
        """
        Returns list of (x, y, w, h) bounding boxes — DNN detector if
        available, Haar cascade otherwise.
        """
        if self.dnn_net is not None:
            return [(x, y, w, h) for x, y, w, h, _ in self._detect_faces_dnn(image_bgr)]
        boxes = self._detect_faces_haar(image_bgr, min_size)
        return [(x, y, w, h) for x, y, w, h, _ in boxes]

    def crop_largest_face(self, image_bgr, padding=0.25):
        """
        Detects faces and returns the best crop. Uses the DNN detector's
        true detection-probability confidence when available (reliable
        across pose/angle); otherwise falls back to Haar cascade ranked
        by its own confidence + a plausible-aspect-ratio filter (helps
        when Haar has multiple candidates including a genuine strong
        frontal-face match, but see the Haar method's docstring for the
        case where this still isn't enough — that's what the DNN path
        above is for).
        Returns None if no face is found.
        """
        H, W = image_bgr.shape[:2]

        if self.dnn_net is not None:
            candidates = self._detect_faces_dnn(image_bgr)
            if not candidates:
                return None
            x, y, w, h, _ = max(candidates, key=lambda c: c[4])
        else:
            candidates = self._detect_faces_haar(image_bgr)
            if not candidates:
                return None
            plausible = [c for c in candidates if 0.7 <= (c[2] / c[3]) <= 1.4]
            pool = plausible if plausible else candidates
            x, y, w, h, _ = max(pool, key=lambda c: (c[4], c[2] * c[3]))

        pad_w, pad_h = int(w * padding), int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(W, x + w + pad_w)
        y2 = min(H, y + h + pad_h)

        return image_bgr[y1:y2, x1:x2]

    def extract_faces_from_video(self, video_path, sample_rate=15, max_frames=30):
        """
        Samples frames from a video at `sample_rate` interval, extracts the
        largest face crop from each sampled frame.
        Returns list of face-crop numpy arrays (BGR).
        """
        cap = cv2.VideoCapture(video_path)
        face_crops = []
        frame_idx = 0

        while cap.isOpened() and len(face_crops) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                crop = self.crop_largest_face(frame)
                if crop is not None and crop.size > 0:
                    face_crops.append(crop)
            frame_idx += 1

        cap.release()
        return face_crops