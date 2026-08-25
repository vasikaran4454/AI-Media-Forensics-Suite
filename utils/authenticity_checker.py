"""
Media Authenticity Checker — non-face tampering detection.

Three independent forensic signals, combined into an overall authenticity score:
1. Error Level Analysis (ELA) — flags regions with inconsistent JPEG
   recompression, a classic sign of localized editing/splicing.
2. Metadata forensics — EXIF inspection for missing data, editing-software
   signatures, or timestamp inconsistencies.
3. Noise consistency — spliced regions often carry different sensor-noise
   statistics than the rest of the image.
"""

import io
import os
import numpy as np
import cv2
from PIL import Image, ExifTags

EDITING_SOFTWARE_SIGNATURES = [
    "photoshop", "gimp", "lightroom", "snapseed", "picsart",
    "facetune", "canva", "pixlr", "affinity photo",
]


# ---------- 1. Error Level Analysis ----------

def error_level_analysis(image_path, quality=90, scale=15):
    """
    Resaves the image at a known JPEG quality and diffs against the original.
    Returns (ela_image_bgr, suspicion_score 0-1).
    Untouched regions of a genuine photo compress fairly uniformly;
    spliced/edited regions often light up brighter/differently in the diff.
    """
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    orig_arr = np.array(original).astype("int16")
    resaved_arr = np.array(resaved).astype("int16")

    diff = np.abs(orig_arr - resaved_arr)
    diff_scaled = np.clip(diff * scale, 0, 255).astype("uint8")

    # Suspicion score: how much variance / hot-spotting is in the ELA map.
    # A uniformly low, even ELA = likely untouched. Bright localized patches
    # (high std relative to mean) = possible localized edit.
    gray_diff = cv2.cvtColor(diff_scaled, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray_diff))
    std_val = float(np.std(gray_diff))
    hot_pixel_ratio = float(np.sum(gray_diff > 60)) / gray_diff.size

    # Heuristic combination — tune thresholds against sample sets before
    # presenting this as a precise number in a viva; frame it as an
    # indicative signal, not a certainty.
    suspicion = min(max((hot_pixel_ratio * 3) + (std_val / 255) * 0.5, 0.0), 1.0)

    ela_bgr = cv2.cvtColor(diff_scaled, cv2.COLOR_RGB2BGR)
    return ela_bgr, suspicion, {"mean": mean_val, "std": std_val, "hot_pixel_ratio": hot_pixel_ratio}


# ---------- 2. Metadata forensics ----------

def metadata_forensics(image_path):
    """
    Inspects EXIF data for red flags:
    - No EXIF at all (common after re-export/screenshot/social media strip —
      not proof of tampering by itself, but reduces provenance confidence)
    - Editing software signature present
    - Modify-date earlier than or equal to original-date inconsistencies
    """
    findings = []
    risk_points = 0
    exif_data = {}

    try:
        img = Image.open(image_path)
        raw_exif = img._getexif() if hasattr(img, "_getexif") else None

        if not raw_exif:
            findings.append("No EXIF metadata found (common after re-export, "
                             "screenshotting, or social media upload — reduces "
                             "provenance confidence but is not proof of tampering).")
            risk_points += 1
        else:
            for tag_id, value in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[str(tag)] = str(value)

            software = exif_data.get("Software", "").lower()
            if software:
                for sig in EDITING_SOFTWARE_SIGNATURES:
                    if sig in software:
                        findings.append(f"Editing software signature detected: '{exif_data.get('Software')}'.")
                        risk_points += 2
                        break

            if "DateTimeOriginal" in exif_data and "DateTime" in exif_data:
                if exif_data["DateTimeOriginal"] != exif_data["DateTime"]:
                    findings.append("Modify-date differs from original capture date — "
                                     "image was likely opened/saved by editing software after capture.")
                    risk_points += 1

            if not findings:
                findings.append("EXIF present, no obvious red flags found.")

    except Exception as e:
        findings.append(f"Could not parse metadata: {e}")

    risk_score = min(risk_points / 4.0, 1.0)
    return findings, risk_score, exif_data


# ---------- 3. Noise consistency analysis ----------

def noise_consistency_analysis(image_path, grid=4):
    """
    Splits the image into a grid, estimates local noise variance per cell
    (via Laplacian), and measures how much the cells deviate from each
    other. Spliced-in regions (from a different source image/sensor)
    often have noticeably different noise variance than the rest.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell_h, cell_w = h // grid, w // grid

    variances = []
    for i in range(grid):
        for j in range(grid):
            cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            if cell.size == 0:
                continue
            lap = cv2.Laplacian(cell, cv2.CV_64F)
            variances.append(float(lap.var()))

    if not variances:
        return 0.0, {}

    variances = np.array(variances)
    mean_v = float(np.mean(variances))
    std_v = float(np.std(variances))
    coeff_of_variation = std_v / (mean_v + 1e-8)

    # High coefficient of variation across grid cells => inconsistent noise
    # texture across the image => possible splice.
    inconsistency_score = min(coeff_of_variation / 2.0, 1.0)
    return inconsistency_score, {"mean_variance": mean_v, "std_variance": std_v}


# ---------- Combined report ----------

def run_full_authenticity_check(image_path):
    ela_img, ela_score, ela_stats = error_level_analysis(image_path)
    meta_findings, meta_score, exif_data = metadata_forensics(image_path)
    noise_score, noise_stats = noise_consistency_analysis(image_path)

    overall_risk = (ela_score * 0.45) + (meta_score * 0.25) + (noise_score * 0.30)
    overall_risk = round(min(max(overall_risk, 0.0), 1.0), 3)

    if overall_risk < 0.3:
        verdict = "Likely Authentic"
    elif overall_risk < 0.6:
        verdict = "Inconclusive — Manual Review Recommended"
    else:
        verdict = "Signs of Possible Tampering"

    return {
        "overall_risk": overall_risk,
        "verdict": verdict,
        "ela": {"score": round(ela_score, 3), "stats": ela_stats, "image": ela_img},
        "metadata": {"score": round(meta_score, 3), "findings": meta_findings, "exif": exif_data},
        "noise": {"score": round(noise_score, 3), "stats": noise_stats},
    }
