"""
Pretrained AI-vs-Real image detector (whole-image signal).

WHY THIS FILE EXISTS: everything else in this project (the GRU deepfake
model, the frequency/noise heuristic) was either built for a different
task (face-swap video deepfakes) or hand-engineered without any labeled
training data. This module wraps a model that was actually TRAINED,
end-to-end, on thousands of labeled real vs AI-generated images — a
categorically stronger signal for "is this whole image AI-generated"
(Midjourney/DALL-E/Stable Diffusion/Flux/etc), which is the actual
question being asked for cases like a fully-AI-generated composite
image (not a real photo with a face-swap applied to it).

Model: Ateeqq/ai-vs-human-image-detector (SigLIP backbone)
https://huggingface.co/Ateeqq/ai-vs-human-image-detector

Requires: torch, transformers (see requirements.txt).
First call downloads ~350MB of weights — needs internet ONCE; cached
under ~/.cache/huggingface afterwards, so subsequent runs are offline
and fast. If torch/transformers aren't installed, or the download
fails (no internet), every function here degrades gracefully to
returning None — callers must treat that as "signal unavailable" and
fall back to the other signals, never as "REAL".
"""

import logging

import cv2

logger = logging.getLogger(__name__)

HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoImageProcessor, SiglipForImageClassification
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning(
        "[ai_detector] torch/transformers not installed — pretrained "
        "AI-image-detector signal unavailable. Install with: "
        "pip install torch transformers --break-system-packages"
    )

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"

_model = None
_processor = None
_load_error = None


def _load():
    """Loads and caches the model. Safe to call repeatedly — no-op after first success."""
    global _model, _processor, _load_error

    if _model is not None:
        return _model
    if not HAS_TRANSFORMERS:
        _load_error = "torch/transformers not installed in this Python environment."
        return None

    try:
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = SiglipForImageClassification.from_pretrained(MODEL_ID)
        _model.eval()
        _load_error = None
        logger.info(f"[ai_detector] Loaded {MODEL_ID}")
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        logger.warning(f"[ai_detector] Failed to load {MODEL_ID}: {e}")
        _model = None

    return _model


def get_load_error():
    return _load_error


def is_available():
    """Cheap check without forcing a (possibly slow, first-time) load."""
    return HAS_TRANSFORMERS


def predict_ai_generated(image_bgr):
    """
    Runs the pretrained SigLIP AI-vs-human classifier on a full image
    (BGR numpy array, e.g. from cv2.imread — NOT just a face crop; this
    model looks at the whole scene, which matters since AI-generation
    artifacts often show up in backgrounds/textures too, not only faces).

    Returns:
        {"fake_probability": float 0-1, "label": "FAKE"/"REAL",
         "model_name": MODEL_ID}
        or None if the model/dependencies are unavailable — callers MUST
        treat None as "no signal", not as evidence of "REAL".
    """
    model = _load()
    if model is None:
        return None

    try:
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        inputs = _processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label = _model.config.id2label
        ai_prob = None
        for idx, label in id2label.items():
            idx = int(idx)
            low = label.lower()
            if "ai" in low or "fake" in low or "artificial" in low or "generated" in low:
                ai_prob = float(probs[idx])
                break
        if ai_prob is None:
            # Label text didn't match any expected pattern — log the
            # actual labels rather than silently guessing an index, so
            # a mislabeled result is loud/debuggable, not silent.
            logger.warning(f"[ai_detector] Unrecognized label set {dict(id2label)} — "
                            f"defaulting to index 0. Verify this is correct for this model version.")
            ai_prob = float(probs[0])

        return {
            "fake_probability": ai_prob,
            "label": "FAKE" if ai_prob >= 0.5 else "REAL",
            "model_name": MODEL_ID,
        }
    except Exception as e:
        global _load_error
        _load_error = f"Inference failed: {type(e).__name__}: {e}"
        logger.warning(f"[ai_detector] Inference failed: {e}")
        return None


__all__ = ["predict_ai_generated", "get_load_error", "is_available"]
