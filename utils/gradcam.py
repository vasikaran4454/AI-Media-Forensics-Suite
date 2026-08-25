"""
Grad-CAM for the CNN-backbone + GRU sequence architecture.

Only meaningful for single-image analysis: the pseudo-sequence is the same
face crop repeated seq_len times, so gradient of the classifier's output
w.r.t. that one frame's backbone conv features correctly reflects that
image's contribution to the prediction. (For genuine multi-frame video,
each frame differs and attributing "the" heatmap to one frame would be
misleading, so Grad-CAM is intentionally not offered for video mode.)
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

HAS_TF = False
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    pass


def _find_last_conv_layer(backbone):
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(classifier_model, face_crop_bgr, input_size):
    """
    Args:
        classifier_model: the loaded GRU classifier (utils.deepfake_model._model)
        face_crop_bgr: BGR face crop, as used for the actual prediction
        input_size: (W, H) the backbone expects — utils.deepfake_model._model_input_size

    Returns a BGR heatmap-overlay image the same size as face_crop_bgr, or
    None if it can't be computed (heuristic mode / missing TF / no conv layer).
    """
    if not HAS_TF or classifier_model is None:
        return None

    try:
        from utils.deepfake_model import _model_config, get_backbone

        if _model_config is None:
            return None

        backbone, preprocess_fn = get_backbone(_model_config["backbone"])
        seq_len = _model_config["seq_len"]

        last_conv_name = _find_last_conv_layer(backbone)
        if last_conv_name is None:
            return None

        conv_submodel = tf.keras.models.Model(
            backbone.inputs, [backbone.get_layer(last_conv_name).output, backbone.output]
        )

        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, input_size).astype("float32")
        batch = preprocess_fn(np.expand_dims(resized, axis=0))
        input_tensor = tf.convert_to_tensor(batch)

        # bool, to match the mask InputLayer's dtype in the rebuilt GRU head
        # (see deepfake_model._build_clean_gru_head) — a float mask would
        # silently fail to match against a bool Input.
        mask = tf.ones((1, seq_len), dtype=tf.bool)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_out, pooled_feat = conv_submodel(input_tensor)
            # Repeat this single frame's feature across the full sequence —
            # matches exactly what predict_face() does at inference time.
            feat_seq = tf.repeat(tf.expand_dims(pooled_feat, axis=1), seq_len, axis=1)
            predictions = classifier_model([feat_seq, mask])
            predictions = tf.reshape(predictions, [-1])
            fake_idx = 1 if predictions.shape[0] >= 2 else 0
            loss = predictions[fake_idx]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            logger.warning("[gradcam] Gradient computation returned None")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (face_crop_bgr.shape[1], face_crop_bgr.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(face_crop_bgr, 0.6, heatmap_color, 0.4, 0)
        return overlay

    except Exception as e:
        logger.warning(f"[gradcam] Failed: {e}")
        return None