"""
Deepfake model loading and prediction.

IMPORTANT ARCHITECTURE NOTE (discovered by inspecting the .h5 files directly):
These are NOT simple image classifiers. They are CNN-feature + GRU sequence
models — each takes:
  1. A sequence of per-frame CNN features (already pooled, not raw pixels):
     - latest_EfficientNetB2.h5 expects (batch, 25, 1408) — EfficientNetB2 pooled features
     - inceptionNet_model.h5    expects (batch, 20, 2048) — InceptionV3 pooled features
  2. A mask tensor (batch, seq_len) marking valid timesteps

So the real pipeline is: face crop(s) -> ImageNet-pretrained CNN backbone
(frozen, NOT fine-tuned — only the GRU head was fine-tuned for deepfake
detection) -> pooled feature vector -> sequence of these -> GRU classifier.

Also fixes: 'time_major' GRU argument was removed in Keras 3 — these .h5
files were saved with an older Keras. We patch around it via a compat GRU
subclass passed as a custom_object during load.
"""

import os
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

HAS_TF = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    HAS_TF = True
except ImportError:
    logger.warning("[deepfake_model] TensorFlow not installed — heuristic mode only")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models")

# Per-model architecture config, confirmed by inspecting each .h5's input layers.
MODEL_CONFIGS = {
    "latest_EfficientNetB2.h5": {
        "seq_len": 25,
        "feat_dim": 1408,
        "backbone": "efficientnetb2",
        "input_size": (260, 260),  # EfficientNetB2's native input resolution
    },
    "inceptionNet_model.h5": {
        "seq_len": 20,
        "feat_dim": 2048,
        "backbone": "inceptionv3",
        "input_size": (299, 299),  # InceptionV3's native input resolution
    },
}
# Try the EfficientNet one first — smaller/faster to run.
MODEL_PRIORITY = ["latest_EfficientNetB2.h5", "inceptionNet_model.h5"]

_model = None
_model_name = None
_model_config = None
_backbones = {}  # cache: backbone name -> (keras_model, preprocess_fn)
_load_error = None  # last exception message, surfaced in the UI for debugging

# Exposed for gradcam.py / app.py — the (W, H) to resize face crops to.
_model_input_size = (260, 260)


# ============================================================
# KERAS 3 COMPATIBILITY — GRU lost the 'time_major' kwarg
# ============================================================
if HAS_TF:
    class CompatGRU(tf.keras.layers.GRU):
        def __init__(self, *args, time_major=None, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)

        def get_config(self):
            config = super().get_config()
            config.pop("time_major", None)
            return config
else:
    CompatGRU = None


# ============================================================
# CLASSIFIER (GRU head) LOADING
# ============================================================
def _build_clean_gru_head(config):
    """
    Rebuilds the GRU classifier head from scratch, matching the exact
    architecture stored in the .h5 model_config (confirmed by inspecting
    the raw JSON in each file):

        seq_input  (None, seq_len, feat_dim) float32
        mask_input (None, seq_len)           bool
        gru(16, return_sequences=True)(seq_input, mask=mask_input)
        gru_1(8, return_sequences=False)
        dropout(0.4)
        dense(8, relu)
        dense_1(2, softmax)

    We build this ourselves instead of using tf.keras.models.load_model()
    because the original file was saved with Keras 2's functional
    serialization, where a layer's `mask` kwarg is stored as a cross-layer
    reference: "inbound_nodes": [[["input_5", 0, 0, {"mask": ["input_6", 0, 0]}]]].
    Keras 3's loader does not resolve this old-style embedded reference —
    it leaves the raw (layer_name, node_index, tensor_index) tuple in
    place instead of the actual mask tensor, which is exactly the
    mask=["'input_6'", '0', '0'] value you saw in the error. No
    custom_objects trick can fix that: the fix is to reconstruct the
    5-layer graph explicitly (so the mask is wired correctly at build
    time) and load only the *weights* from the file by layer name,
    bypassing the broken config resolution entirely.
    """
    seq_input = tf.keras.Input(shape=(config["seq_len"], config["feat_dim"]),
                                dtype="float32", name="seq_input")
    mask_input = tf.keras.Input(shape=(config["seq_len"],),
                                 dtype="bool", name="mask_input")

    x = tf.keras.layers.GRU(16, return_sequences=True, name="gru")(seq_input, mask=mask_input)
    x = tf.keras.layers.GRU(8, return_sequences=False, name="gru_1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense")(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="dense_1")(x)

    return tf.keras.Model(inputs=[seq_input, mask_input], outputs=out, name="gru_head_clean")


def load_model():
    """
    Loads the fine-tuned GRU classifier head (cached after first call).
    Sets global _model_config / _model_input_size to match whichever
    model file loaded successfully.
    """
    global _model, _model_name, _model_config, _model_input_size, _load_error

    if _model is not None:
        return _model

    if not HAS_TF:
        _load_error = "TensorFlow is not installed in this Python environment."
        return None

    errors = []
    for model_file in MODEL_PRIORITY:
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            errors.append(f"{model_file}: file not found at {model_path}")
            continue

        config = MODEL_CONFIGS[model_file]

        # Rebuild the architecture ourselves (see _build_clean_gru_head
        # docstring) instead of tf.keras.models.load_model(), which
        # cannot resolve this file's legacy mask-wiring correctly.
        try:
            model = _build_clean_gru_head(config)
            model.load_weights(model_path, by_name=True, skip_mismatch=False)
        except Exception as e:
            errors.append(f"{model_file}: clean-rebuild load_weights failed: {type(e).__name__}: {e}")
            logger.warning(f"[deepfake_model] Clean rebuild failed for {model_file}: {e}")

            # Last-resort fallback: try the old full-model loader in case
            # a differently-saved file doesn't have the mask-reference issue.
            try:
                model = tf.keras.models.load_model(
                    model_path, custom_objects={"GRU": CompatGRU}, compile=False
                )
            except Exception as e2:
                errors.append(f"{model_file}: fallback load_model also failed: {type(e2).__name__}: {e2}")
                continue

        _model = model
        _model_name = model_file
        _model_config = config
        _model_input_size = config["input_size"]
        _load_error = None
        logger.info(f"[deepfake_model] Loaded {model_file} (clean rebuild + weights-by-name)")
        return _model

    _load_error = " | ".join(errors) if errors else "Unknown failure"
    logger.warning("[deepfake_model] All models failed to load — heuristic mode")
    return None


def get_model_info():
    if _model is None:
        return {"loaded": False, "error": _load_error}
    return {
        "loaded": True,
        "name": _model_name,
        "seq_len": _model_config["seq_len"],
        "feat_dim": _model_config["feat_dim"],
        "backbone": _model_config["backbone"],
    }


def get_load_error():
    return _load_error


# ============================================================
# BACKBONE (frozen ImageNet feature extractor) — lazy-loaded
# ============================================================
def get_backbone(backbone_name):
    """
    Returns (keras_model, preprocess_fn) for the given backbone, building
    and caching it on first use. Requires internet on first run (downloads
    ImageNet weights, ~35-90MB, cached by Keras under ~/.keras/models).
    """
    if backbone_name in _backbones:
        return _backbones[backbone_name]

    if backbone_name == "efficientnetb2":
        from tensorflow.keras.applications import efficientnet
        model = tf.keras.applications.EfficientNetB2(
            weights="imagenet", include_top=False, pooling="avg"
        )
        preprocess_fn = efficientnet.preprocess_input
    elif backbone_name == "inceptionv3":
        from tensorflow.keras.applications import inception_v3
        model = tf.keras.applications.InceptionV3(
            weights="imagenet", include_top=False, pooling="avg"
        )
        preprocess_fn = inception_v3.preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    _backbones[backbone_name] = (model, preprocess_fn)
    return model, preprocess_fn


def _extract_features(face_crops_bgr, config):
    """
    Runs the frozen CNN backbone over a batch of face crops (BGR numpy
    arrays) and returns pooled features, shape (N, feat_dim).
    """
    backbone, preprocess_fn = get_backbone(config["backbone"])
    size = config["input_size"]

    batch = []
    for crop in face_crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, size)
        batch.append(resized.astype("float32"))
    batch = np.stack(batch, axis=0)
    batch = preprocess_fn(batch)

    features = backbone.predict(batch, verbose=0)
    return features  # (N, feat_dim)


def _build_sequence(face_crops_bgr, config):
    """
    Pads/samples a list of face crops to exactly seq_len entries, extracts
    features, and returns (feat_seq, mask) ready for the classifier:
      feat_seq: (1, seq_len, feat_dim)
      mask:     (1, seq_len)
    """
    seq_len = config["seq_len"]
    frames = list(face_crops_bgr)

    if len(frames) == 0:
        return None, None

    if len(frames) >= seq_len:
        # Evenly sample seq_len frames across the full range.
        idx = np.linspace(0, len(frames) - 1, seq_len).astype(int)
        frames = [frames[i] for i in idx]
    else:
        # Pad by repeating the last frame.
        frames = frames + [frames[-1]] * (seq_len - len(frames))

    features = _extract_features(frames, config)  # (seq_len, feat_dim)
    feat_seq = np.expand_dims(features, axis=0)  # (1, seq_len, feat_dim)
    # Must be bool, matching the mask InputLayer's dtype in the original
    # model ("input_6"/"input_4" were saved as dtype=bool) — the GRU's
    # mask= argument expects a boolean mask, not float.
    mask = np.ones((1, seq_len), dtype="bool")
    return feat_seq, mask


def _run_classifier(feat_seq, mask):
    """Runs the GRU classifier on one prepared sequence. Returns fake_probability."""
    pred = _model.predict([feat_seq, mask], verbose=0)
    pred = np.array(pred).flatten()
    # 2-class softmax output; convention in this checkpoint is
    # [P(real), P(fake)] based on the training pipeline this was adapted from.
    fake_prob = float(pred[1]) if pred.shape[0] >= 2 else float(pred[0])
    return fake_prob


# ============================================================
# PUBLIC PREDICTION API
# ============================================================
def predict_face(face_crop_bgr, full_image_bgr=None):
    """
    Combines up to THREE independent signals, in order of trust:

      1. pretrained_prob  - a genuinely-trained AI-vs-real image
                             classifier (utils/ai_detector.py), run on
                             the FULL original image when available
                             (full_image_bgr) since AI-generation
                             artifacts show up scene-wide, not just on
                             faces. This is the strongest signal here —
                             it was actually trained on thousands of
                             labeled real/AI images — so it dominates
                             the ensemble weighting when available.
      2. model_prob       - the GRU deepfake model (weak on a single
                             repeated frame — see docstring history in
                             this file; kept as a secondary vote).
      3. heuristic_prob   - frequency/noise/texture heuristic on the
                             face crop (see _heuristic_predict).

    If the pretrained detector isn't installed/available, this
    transparently falls back to the (model + heuristic) ensemble only —
    still functional, just weaker, exactly as before.
    """
    model = load_model()
    heuristic_result = _heuristic_predict(face_crop_bgr)
    heuristic_prob = heuristic_result["fake_probability"]

    pretrained_result = None
    try:
        from utils.ai_detector import predict_ai_generated
        pretrained_result = predict_ai_generated(full_image_bgr if full_image_bgr is not None else face_crop_bgr)
    except Exception as e:
        logger.warning(f"[deepfake_model] Pretrained AI-detector unavailable: {e}")

    model_prob = None
    if model is not None:
        try:
            feat_seq, mask = _build_sequence([face_crop_bgr], _model_config)
            model_prob = _run_classifier(feat_seq, mask)
        except Exception as e:
            global _load_error
            _load_error = f"Prediction failed (model loaded OK, likely backbone issue): {type(e).__name__}: {e}"
            logger.warning(f"[deepfake_model] Model prediction failed: {e}")

    # ---- Case 1: pretrained detector available — it dominates ----
    if pretrained_result is not None:
        pretrained_prob = pretrained_result["fake_probability"]

        if model_prob is not None:
            secondary_prob = 0.6 * model_prob + 0.4 * heuristic_prob
        else:
            secondary_prob = heuristic_prob

        fake_prob = 0.7 * pretrained_prob + 0.3 * secondary_prob

        return {
            "label": "FAKE" if fake_prob >= 0.5 else "REAL",
            "fake_probability": float(np.clip(fake_prob, 0.01, 0.99)),
            "mode": "ensemble_pretrained",
            "model_name": pretrained_result["model_name"],
            "pretrained_probability": round(float(pretrained_prob), 3),
            "model_probability": round(float(model_prob), 3) if model_prob is not None else None,
            "heuristic_probability": round(float(heuristic_prob), 3),
            "note": "Primary signal: a pretrained AI-vs-real image classifier "
                    "(trained on labeled real/AI-generated images — see "
                    "pretrained_probability), weighted 70%. Secondary signal: "
                    "the video-native GRU model and frequency/noise heuristic "
                    "(30% combined), included as a cross-check.",
        }

    # ---- Case 2: no pretrained detector — fall back to prior ensemble ----
    if model_prob is not None:
        model_confidence = abs(model_prob - 0.5) * 2
        heuristic_weight = 0.35 + (1 - model_confidence) * 0.35
        model_weight = 1 - heuristic_weight
        fake_prob = model_weight * model_prob + heuristic_weight * heuristic_prob

        return {
            "label": "FAKE" if fake_prob >= 0.5 else "REAL",
            "fake_probability": float(np.clip(fake_prob, 0.01, 0.99)),
            "mode": "ensemble",
            "model_name": _model_name,
            "model_probability": round(float(model_prob), 3),
            "heuristic_probability": round(float(heuristic_prob), 3),
            "model_weight": round(float(model_weight), 2),
            "heuristic_weight": round(float(heuristic_weight), 2),
            "note": "Pretrained AI-detector unavailable (install torch+transformers "
                    "for a stronger signal — see utils/ai_detector.py). Falling back "
                    "to GRU model + frequency/noise heuristic only.",
        }

    return heuristic_result


def predict_video(face_crops_bgr):
    """
    Video: if more crops are available than the model's seq_len, runs
    several overlapping windows across the full sampled range and averages
    them — giving a genuine temporal breakdown rather than one flat number.

    Also runs the pretrained AI-image detector (see ai_detector.py) on
    one representative frame, as a cross-check signal — this catches
    fully-AI-generated video (e.g. a Sora/Runway-style clip) that the
    face-swap-focused GRU model wasn't trained to recognize as fake.
    """
    model = load_model()

    pretrained_result = None
    if face_crops_bgr:
        try:
            from utils.ai_detector import predict_ai_generated
            pretrained_result = predict_ai_generated(face_crops_bgr[len(face_crops_bgr) // 2])
        except Exception as e:
            logger.warning(f"[deepfake_model] Pretrained AI-detector unavailable for video: {e}")

    if model is not None and len(face_crops_bgr) >= 1:
        try:
            seq_len = _model_config["seq_len"]
            n = len(face_crops_bgr)

            if n <= seq_len:
                feat_seq, mask = _build_sequence(face_crops_bgr, _model_config)
                fake_prob = _run_classifier(feat_seq, mask)
                window_scores = [fake_prob]
            else:
                n_windows = min(6, n - seq_len + 1)
                starts = np.linspace(0, n - seq_len, n_windows).astype(int)
                starts = sorted(set(starts.tolist()))
                window_scores = []
                for s in starts:
                    window = face_crops_bgr[s:s + seq_len]
                    feat_seq, mask = _build_sequence(window, _model_config)
                    window_scores.append(_run_classifier(feat_seq, mask))
                fake_prob = float(np.mean(window_scores))

            if pretrained_result is not None:
                combined_prob = 0.6 * pretrained_result["fake_probability"] + 0.4 * fake_prob
            else:
                combined_prob = fake_prob

            windows_flagged = sum(1 for s in window_scores if s >= 0.5)

            result = {
                "label": "FAKE" if combined_prob >= 0.5 else "REAL",
                "fake_probability": float(np.clip(combined_prob, 0.01, 0.99)),
                "mode": "ensemble_pretrained" if pretrained_result is not None else "model",
                "model_name": _model_name,
                "frames_analyzed": len(face_crops_bgr),
                "frames_flagged_fake": windows_flagged,
                "per_frame_scores": window_scores,
                "windowed": True,
                "gru_probability": round(float(fake_prob), 3),
            }
            if pretrained_result is not None:
                result["pretrained_probability"] = round(float(pretrained_result["fake_probability"]), 3)
                result["note"] = ("Combines the video-native GRU deepfake model (40% weight, "
                                   "see gru_probability) with a pretrained whole-image AI-generation "
                                   "classifier run on a representative frame (60% weight, see "
                                   "pretrained_probability) — the latter catches fully-AI-generated "
                                   "video that a face-swap-focused model alone would miss.")
            return result
        except Exception as e:
            global _load_error
            _load_error = f"Video prediction failed: {type(e).__name__}: {e}"
            logger.warning(f"[deepfake_model] Video prediction failed: {e}")

    # Heuristic fallback (GRU unavailable) — still use pretrained detector if we have it
    per_frame = [_heuristic_predict(fc) for fc in face_crops_bgr]
    fake_probs = [r["fake_probability"] for r in per_frame]
    avg_prob = float(np.mean(fake_probs)) if fake_probs else 0.5

    if pretrained_result is not None:
        avg_prob = 0.7 * pretrained_result["fake_probability"] + 0.3 * avg_prob

    return {
        "label": "FAKE" if avg_prob >= 0.5 else "REAL",
        "fake_probability": avg_prob,
        "mode": "ensemble_pretrained" if pretrained_result is not None else "heuristic",
        "model_name": pretrained_result["model_name"] if pretrained_result is not None else "frequency_domain_heuristic",
        "frames_analyzed": len(face_crops_bgr),
        "frames_flagged_fake": sum(1 for p in fake_probs if p >= 0.5),
        "per_frame_scores": fake_probs,
        "windowed": False,
    }


# ============================================================
# HEURISTIC FALLBACK (only used if TF/models are unavailable)
# Balanced multi-signal score — not biased toward always-FAKE.
# ============================================================
def _heuristic_predict(face_crop_bgr):
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY) if len(face_crop_bgr.shape) == 3 else face_crop_bgr.copy()
    gray = cv2.resize(gray, (128, 128)).astype(np.float32)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2

    low_mask = np.zeros_like(magnitude, dtype=bool)
    low_mask[ch - h // 8:ch + h // 8, cw - w // 8:cw + w // 8] = True
    low_energy = np.mean(magnitude[low_mask])

    high_mask = ~low_mask
    border = 8
    high_mask[:border, :] = False
    high_mask[-border:, :] = False
    high_mask[:, :border] = False
    high_mask[:, -border:] = False
    high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 1
    hf_ratio = high_energy / (low_energy + 1e-10)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_var = np.var(gray - blurred)

    edges_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(edges_x ** 2 + edges_y ** 2)
    edge_cv = np.std(edge_mag) / (np.mean(edge_mag) + 1e-10)

    # ---- Skin-texture micro-detail signal (targets AI-generation directly) ----
    # Real camera photos carry fine pore/skin-texture noise that GAN/diffusion
    # generators characteristically over-smooth (they optimize for perceptual
    # sharpness, not sensor-level micro-texture). We measure this in a central
    # cheek/forehead-ish patch, away from eyes/mouth edges which have strong
    # "real" edges of their own that would otherwise mask the effect.
    patch = gray[int(0.30 * 128):int(0.70 * 128), int(0.30 * 128):int(0.70 * 128)]
    # cv2.Laplacian with a float32 source + CV_64F destination hits an
    # unsupported AVX2 codepath on some OpenCV builds
    # ("Unsupported combination of source format (=5) and destination
    # format (=6)"). Cast to uint8 first — matches how Laplacian is
    # already used elsewhere in this codebase (noise_consistency_analysis
    # in authenticity_checker.py, which never hit this because it starts
    # from a uint8 image) — and avoids the unsupported combination.
    micro_lap = cv2.Laplacian(patch.astype(np.uint8), cv2.CV_64F)
    micro_texture_var = float(np.var(micro_lap))

    score = 0.45
    if hf_ratio > 2.0:
        score += 0.15
    elif hf_ratio < 0.05:
        score += 0.10
    elif 0.1 < hf_ratio < 0.8:
        score -= 0.10

    if noise_var < 10:
        score += 0.15
    elif 30 < noise_var < 200:
        score -= 0.10

    if edge_cv < 0.3:
        score += 0.10
    elif edge_cv > 0.5:
        score -= 0.05

    # Low micro-texture variance = unnaturally smooth skin = AI-generation tell.
    if micro_texture_var < 15:
        score += 0.15
    elif micro_texture_var > 60:
        score -= 0.08

    score = float(np.clip(score, 0.08, 0.92))
    return {
        "label": "FAKE" if score >= 0.5 else "REAL",
        "fake_probability": score,
        "mode": "heuristic",
        "model_name": "frequency_domain_heuristic",
        "signals": {
            "hf_ratio": round(float(hf_ratio), 3),
            "noise_var": round(float(noise_var), 3),
            "edge_cv": round(float(edge_cv), 3),
            "micro_texture_var": round(micro_texture_var, 3),
        },
    }


__all__ = ["load_model", "predict_face", "predict_video", "get_model_info", "get_backbone", "get_load_error"]