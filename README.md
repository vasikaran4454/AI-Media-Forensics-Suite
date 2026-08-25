# AI Media Forensics Suite

Academic showcase project — single-page toolkit with two tabs:

1. **Deepfake Detection** — CNN-based (EfficientNetB2 / InceptionNet) classification of
   faces in images/video as REAL or FAKE, with Grad-CAM explainability heatmaps and
   frame-by-frame video analysis.
2. **Media Authenticity Checker** — non-face tampering detection via Error Level
   Analysis (ELA), EXIF metadata forensics, and noise-consistency analysis.

Both tabs produce a downloadable PDF report.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows:
```bash
python -m streamlit run app.py
```

## Project structure

```
app.py                     # Single-page app — both modules as tabs
utils/
  face_utils.py            # OpenCV face detection (image + video)
  deepfake_model.py        # Model loading + prediction (model or heuristic fallback)
  gradcam.py                # Explainability heatmaps
  authenticity_checker.py   # ELA + metadata + noise analysis
  report_generator.py       # ReportLab PDF report
  theme.py                   # Branding — change PROJECT_NAME in theme.py to rename instantly
assets/models/               # Pretrained .h5 weights
```

## Rebranding

To rename the project, edit one line: `PROJECT_NAME` at the top of `utils/theme.py`.

## Honesty notes (for viva / report)

- If the `.h5` model can't be loaded on a given machine (missing TensorFlow, or
  input-shape mismatch), the tool **automatically falls back to a frequency-domain
  heuristic** and clearly labels output as "heuristic mode" — never silently presented
  as the trained model's result.
- ELA / metadata / noise scores in the Authenticity Checker are indicative signals,
  not forensic-grade certainty — the report and UI say this explicitly.
