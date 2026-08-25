import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import cv2
import tempfile
import time
from datetime import datetime

from utils.theme import CUSTOM_CSS, BRAND_HEADER_HTML, PROJECT_NAME, risk_color
from utils.face_utils import FaceDetector
from utils.deepfake_model import predict_face, predict_video, load_model
from utils.gradcam import generate_gradcam
from utils.authenticity_checker import run_full_authenticity_check
from utils.report_generator import build_report, get_report_temp_path

st.set_page_config(page_title=PROJECT_NAME, page_icon="🛡️", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Auto-scroll to top on every rerun — otherwise clicking "Continue" leaves
# the user scrolled down at the old button position instead of seeing the
# new step's content.
st.components.v1.html(
    """<script>
    window.parent.document.querySelector('section.main')?.scrollTo({top: 0, behavior: 'smooth'});
    </script>""",
    height=0,
)


def _assess(s):
    if s < 0.3: return "✓ Clean"
    if s < 0.6: return "⚠ Suspicious"
    return "✕ Anomalous"


def _safe_img(uploader):
    if uploader is None: return None
    try:
        uploader.seek(0)
        return cv2.imdecode(np.asarray(bytearray(uploader.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _det(): return FaceDetector()

@st.cache_resource(show_spinner=False)
def _mdl(): return load_model()

detector = _det()

# Init state
if "page" not in st.session_state:
    st.session_state.page = "welcome"
for k in ["step","df_last","df_orig","df_res","df_crop","df_gcam",
          "au_step","au_last","au_orig","au_res","au_ip","au_fn"]:
    if k not in st.session_state:
        st.session_state[k] = 1 if k in ("step","au_step") else None


# ═══════════════════════════════════════
# WELCOME PAGE
# ═══════════════════════════════════════
if st.session_state.page == "welcome":
    st.markdown('''
    <div class="welcome-page">
        <div class="welcome-bg-orb welcome-orb-1"></div>
        <div class="welcome-bg-orb welcome-orb-2"></div>
        <div class="welcome-badge">FORENSIC-GRADE ANALYSIS</div>
        <h1 class="welcome-title">Detect Deepfakes<br><span class="welcome-gradient">Before They Deceive</span></h1>
        <p class="welcome-subtitle">AI-powered media authentication combining deep-learning classification with classical forensic analysis. All processing runs locally on your machine.</p>
        <div class="welcome-features">
            <div class="wf-card"><div class="wf-icon">🎭</div><div class="wf-text"><div class="wf-name">Deepfake Detection</div><div class="wf-desc">CNN classifier with Grad-CAM explainability</div></div></div>
            <div class="wf-card"><div class="wf-icon">🔎</div><div class="wf-text"><div class="wf-name">Media Authenticity</div><div class="wf-desc">ELA, EXIF forensics & noise analysis</div></div></div>
            <div class="wf-card"><div class="wf-icon">📄</div><div class="wf-text"><div class="wf-name">Forensic Reports</div><div class="wf-desc">Certified PDF with case documentation</div></div></div>
        </div>
        <div class="welcome-stats">
            <div class="ws-item"><div class="ws-val">7</div><div class="ws-label">Modules</div></div>
            <div class="ws-divider"></div>
            <div class="ws-item"><div class="ws-val">100%</div><div class="ws-label">Local</div></div>
            <div class="ws-divider"></div>
            <div class="ws-item"><div class="ws-val">&lt;5s</div><div class="ws-label">Speed</div></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if st.button("Enter Forensic Lab  →", type="primary", key="enter_lab", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()


# ═══════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════
elif st.session_state.page == "dashboard":

    st.markdown(BRAND_HEADER_HTML.format(
        title=PROJECT_NAME,
        subtitle="Forensic-grade media authentication — deep-learning classification with classical forensic analysis."
    ), unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎭  Deepfake Detection", "🔎  Media Authenticity Checker"])

    # ─────────── DEEPFAKE TAB ───────────
    with tab1:
        step = st.session_state.step

        st.markdown(f'''
        <div class="step-wizard">
            <div class="sw-step {"active" if step>=1 else ""} {"done" if step>1 else ""}"><div class="sw-num">1</div><div class="sw-label">Upload</div></div>
            <div class="sw-line {"active" if step>=2 else ""}"></div>
            <div class="sw-step {"active" if step>=2 else ""} {"done" if step>2 else ""}"><div class="sw-num">2</div><div class="sw-label">Analyze</div></div>
            <div class="sw-line {"active" if step>=3 else ""}"></div>
            <div class="sw-step {"active" if step>=3 else ""} {"done" if step>3 else ""}"><div class="sw-num">3</div><div class="sw-label">Results</div></div>
            <div class="sw-line {"active" if step>=4 else ""}"></div>
            <div class="sw-step {"active" if step>=4 else ""}"><div class="sw-num">4</div><div class="sw-label">Report</div></div>
        </div>''', unsafe_allow_html=True)

        # ── STEP 1: UPLOAD ──
        if step == 1:
            st.markdown('<div class="step-header"><div class="sh-num">01</div><div><div class="sh-title">Upload Evidence</div><div class="sh-desc">Select an image or video containing a face.</div></div></div>', unsafe_allow_html=True)

            c1, c2 = st.columns([1, 5])
            with c1:
                file_type = st.radio("Type", ["Image", "Video"], key="df_type", label_visibility="collapsed")
            with c2:
                up = st.file_uploader("Upload file", type=["jpg","jpeg","png"] if file_type=="Image" else ["mp4","mov","avi"], key="df_up", label_visibility="collapsed")

            if up is not None:
                # Read and store image NOW — never re-read later
                fb = np.asarray(bytearray(up.read()), dtype=np.uint8)
                # Store the chosen type explicitly at upload time, into its
                # OWN session_state key (df_media_type) — separate from the
                # radio widget's own key (df_type). Relying on a widget's
                # key-based session_state value once that widget is no
                # longer being rendered (e.g. after moving to Step 2, where
                # this radio isn't drawn) is fragile across Streamlit
                # reruns; storing our own explicit copy here removes that
                # whole class of bug (confirmed case: Step 2 read stale/
                # default "Image" and showed "No image loaded" even when
                # "Video" was clearly selected and a video was uploaded).
                st.session_state.df_media_type = file_type
                if file_type == "Image":
                    img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                    if img is not None:
                        st.session_state.df_orig = img
                        st.session_state.df_last = up.name
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.session_state.df_last = up.name
                    st.session_state._video_bytes = fb
                    st.video(up)

                if st.button("Continue to Analysis  →", type="primary", key="df_next1", use_container_width=True):
                    st.session_state.step = 2
                    st.rerun()

        # ── STEP 2: ANALYZE ──
        elif step == 2:
            st.markdown('<div class="step-header"><div class="sh-num">02</div><div><div class="sh-title">Run Analysis</div><div class="sh-desc">Click below to start forensic examination.</div></div></div>', unsafe_allow_html=True)

            # Show preview from stored image
            if st.session_state.df_orig is not None:
                st.image(cv2.cvtColor(st.session_state.df_orig, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Uploaded evidence")

            if st.button("🔍  Run Forensic Analysis", type="primary", key="df_run", use_container_width=True):
                t0 = time.time()
                file_type = st.session_state.get("df_media_type", "Image")

                if file_type == "Image":
                    img = st.session_state.df_orig
                    if img is None:
                        st.error("No image loaded. Go back and upload again.")
                    else:
                        with st.status("Running analysis...", expanded=True) as s:
                            s.write("🔎 Detecting face regions...")
                            crop = detector.crop_largest_face(img)
                            if crop is None:
                                s.update(label="No face found", state="error")
                                st.error("No face detected. Try a clearer front-facing photo.")
                            else:
                                s.write("🧠 Running CNN classifier...")
                                res = predict_face(crop, full_image_bgr=img)
                                s.write("🗺️  Generating Grad-CAM...")
                                mdl = _mdl()
                                hm = None
                                if res["mode"] in ("model", "ensemble", "ensemble_pretrained") and mdl is not None:
                                    from utils.deepfake_model import _model_input_size
                                    hm = generate_gradcam(mdl, crop, _model_input_size)
                                res["time"] = round(time.time() - t0, 1)
                                st.session_state.df_res = res
                                st.session_state.df_crop = crop
                                st.session_state.df_gcam = hm
                                s.update(label=f"Complete — {res['time']}s", state="complete")
                                time.sleep(0.5)
                                st.session_state.step = 3
                                st.rerun()
                else:
                    # Video
                    fb = st.session_state.get("_video_bytes")
                    if fb is None:
                        st.error("No video loaded. Go back and upload again.")
                    else:
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tf.write(fb)
                        vp = tf.name
                        with st.status("Running analysis...", expanded=True) as s:
                            s.write("🎬 Sampling frames & detecting faces...")
                            crops = detector.extract_faces_from_video(vp, sample_rate=15, max_frames=30)
                            if not crops:
                                s.update(label="No faces found", state="error")
                                st.error("No faces detected.")
                            else:
                                s.write(f"🧠 Classifying {len(crops)} crops...")
                                res = predict_video(crops)
                                res["time"] = round(time.time() - t0, 1)
                                st.session_state.df_res = res
                                st.session_state.df_crop = crops[0]
                                st.session_state.df_gcam = None
                                try: os.unlink(vp)
                                except: pass
                                s.update(label=f"Complete — {res['time']}s", state="complete")
                                time.sleep(0.5)
                                st.session_state.step = 3
                                st.rerun()

            if st.button("←  Back", key="df_back2"):
                st.session_state.step = 1
                st.rerun()

        # ── STEP 3: RESULTS ──
        elif step == 3:
            st.markdown('<div class="step-header"><div class="sh-num">03</div><div><div class="sh-title">Examination Results</div><div class="sh-desc">Forensic findings for your evidence.</div></div></div>', unsafe_allow_html=True)

            r = st.session_state.df_res
            if r:
                c = risk_color(r["fake_probability"])
                p = r["fake_probability"] * 100
                conf = "HIGH" if p>80 or p<20 else "MODERATE" if p>60 or p<40 else "LOW"

                # Build the optional "Model" tag as a single string BEFORE
                # embedding it — leaving a blank line inside the raw HTML
                # (when the conditional is empty) causes Streamlit's markdown
                # parser to terminate the HTML block early, which made the
                # closing </div> tags render as literal visible text.
                model_tag_html = (
                    f'<span class="vb-tag">Model: <b>{r["model_name"]}</b></span>'
                    if r["mode"] in ("model", "ensemble", "ensemble_pretrained") else ""
                )
                vb_meta_html = (
                    f'<span class="vb-tag">Confidence: <b>{conf}</b></span>'
                    f'<span class="vb-tag">Mode: <b>{r["mode"].upper()}</b></span>'
                    f'{model_tag_html}'
                )

                st.markdown(f'''
                <div class="verdict-banner" style="border-color:{c}">
                    <div class="vb-row">
                        <span class="vb-label" style="color:{c}">{r["label"]}</span>
                        <span class="vb-time"><span class="vb-time-icon">⏱</span> {r.get("time","?")}s</span>
                    </div>
                    <div class="vb-prob">{p:.1f}% fake probability</div>
                    <div class="vb-bar-wrap"><div class="vb-bar" style="width:{p}%;background:{c}"></div></div>
                    <div class="vb-meta">{vb_meta_html}</div>
                </div>''', unsafe_allow_html=True)

                if r["mode"] == "heuristic":
                    from utils.deepfake_model import get_load_error
                    err = get_load_error()
                    st.warning("⚠️ **Heuristic mode** — model weights unavailable. Frequency-domain signal only.")
                    if err:
                        with st.expander("Why heuristic mode? (technical details)"):
                            st.code(err, language=None)
                elif r["mode"] == "ensemble_pretrained":
                    st.success("✅ **Pretrained AI-detector active** — primary signal.")
                    st.info(f"ℹ️ {r['note']}")
                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        st.metric("Pretrained AI-detector (primary)", f"{r['pretrained_probability']*100:.1f}% fake",
                                   help="Weight: 70% — trained on labeled real/AI-generated images")
                    with bc2:
                        if r.get("model_probability") is not None:
                            st.metric("GRU model (secondary)", f"{r['model_probability']*100:.1f}% fake",
                                       help="Weight: 30% × 60% = 18%")
                        else:
                            st.metric("GRU model (secondary)", "unavailable")
                    with bc3:
                        st.metric("Heuristic (secondary)", f"{r['heuristic_probability']*100:.1f}% fake",
                                   help="Weight: 30% × 40% = 12% (or 30% if GRU unavailable)")
                elif r["mode"] == "ensemble":
                    st.info(f"ℹ️ {r['note']}")
                    from utils.ai_detector import is_available as _ai_detector_installed
                    if not _ai_detector_installed():
                        st.caption("💡 Install `torch` + `transformers` for a much stronger pretrained AI-detection signal — see utils/ai_detector.py docstring.")
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.metric("Model signal (video-native CNN+GRU)", f"{r['model_probability']*100:.1f}% fake",
                                   help=f"Weight in final score: {r['model_weight']*100:.0f}%")
                    with bc2:
                        st.metric("Heuristic signal (frequency/noise/texture)", f"{r['heuristic_probability']*100:.1f}% fake",
                                   help=f"Weight in final score: {r['heuristic_weight']*100:.0f}%")
                elif r.get("note"):
                    st.info(f"ℹ️ {r['note']}")

                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown('<div class="evidence-label">01 — Detected Face</div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(st.session_state.df_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                with ic2:
                    if st.session_state.df_gcam is not None:
                        st.markdown('<div class="evidence-label">02 — Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                        st.image(cv2.cvtColor(st.session_state.df_gcam, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.markdown('<div class="evidence-label">02 — Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                        st.info("Unavailable in heuristic mode or video.")

                if "debug" in r:
                    with st.expander("Technical Details"): st.json(r["debug"])

                if "frames_analyzed" in r:
                    st.markdown('<div class="img-card-title" style="margin-top:1.5rem"><span class="ict-dot"></span>Temporal Breakdown</div>', unsafe_allow_html=True)
                    st.line_chart(r["per_frame_scores"])
                    fl=r["frames_flagged_fake"]; ft=r["frames_analyzed"]
                    if r.get("windowed"):
                        st.caption(f"**{len(r['per_frame_scores'])}** overlapping sequence windows sampled across **{ft}** detected face frames — "
                                   f"**{fl}** window(s) scored FAKE. (This architecture evaluates a full frame sequence "
                                   f"jointly, not independent single frames.)")
                    else:
                        st.caption(f"**{fl}** / **{ft}** windows flagged ({fl/max(1,ft)*100:.0f}%)")

                c_b, c_n = st.columns([1, 2])
                with c_b:
                    if st.button("←  Re-analyze", key="df_back3"):
                        st.session_state.step = 1
                        for k in ["df_res","df_crop","df_gcam","df_orig","_video_bytes","df_media_type"]:
                            st.session_state[k] = None
                        st.rerun()
                with c_n:
                    if st.button("Download Report  →", type="primary", key="df_next3", use_container_width=True):
                        st.session_state.step = 4
                        st.rerun()

        # ── STEP 4: REPORT ──
        elif step == 4:
            st.markdown('<div class="step-header"><div class="sh-num">04</div><div><div class="sh-title">Forensic Report</div><div class="sh-desc">Download your certified examination report.</div></div></div>', unsafe_allow_html=True)

            st.markdown('''
            <div class="pdf-hero">
                <div class="pdf-hero-icon">📄</div>
                <div class="pdf-hero-text">
                    <div class="pdf-hero-title">Forensic Examination Report</div>
                    <div class="pdf-hero-desc">Includes original image, face crop, heatmap, metrics, methodology & certification.</div>
                </div>
            </div>
            <div class="report-features">
                <div class="rf-item"><span class="rf-check">✓</span> Original evidence image</div>
                <div class="rf-item"><span class="rf-check">✓</span> Face crop & Grad-CAM</div>
                <div class="rf-item"><span class="rf-check">✓</span> Detailed metrics table</div>
                <div class="rf-item"><span class="rf-check">✓</span> Methodology documentation</div>
                <div class="rf-item"><span class="rf-check">✓</span> Case ID & certification</div>
                <div class="rf-item"><span class="rf-check">✓</span> CONFIDENTIAL watermark</div>
            </div>''', unsafe_allow_html=True)

            r = st.session_state.df_res
            if r:
                orig = st.session_state.df_orig
                imgs = {"original": orig, "face_crop": st.session_state.df_crop, "gradcam": st.session_state.df_gcam}
                op = get_report_temp_path("deepfake_report.pdf")
                build_report(op, st.session_state.df_last or "uploaded", "deepfake", r, images=imgs)
                with open(op, "rb") as f:
                    st.download_button("⬇️  Download Forensic Report (PDF)", f,
                        file_name=f"forensic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", key="df_dl", use_container_width=True)

            if st.button("←  Analyze Another File", key="df_back4"):
                st.session_state.step = 1
                for k in ["df_res","df_crop","df_gcam","df_orig","df_last","_video_bytes","df_media_type"]:
                    st.session_state[k] = None
                st.rerun()

    # ─────────── AUTHENTICITY TAB ───────────
    with tab2:
        sa = st.session_state.au_step

        st.markdown(f'''
        <div class="step-wizard">
            <div class="sw-step {"active" if sa>=1 else ""} {"done" if sa>1 else ""}"><div class="sw-num">1</div><div class="sw-label">Upload</div></div>
            <div class="sw-line {"active" if sa>=2 else ""}"></div>
            <div class="sw-step {"active" if sa>=2 else ""} {"done" if sa>2 else ""}"><div class="sw-num">2</div><div class="sw-label">Analyze</div></div>
            <div class="sw-line {"active" if sa>=3 else ""}"></div>
            <div class="sw-step {"active" if sa>=3 else ""} {"done" if sa>3 else ""}"><div class="sw-num">3</div><div class="sw-label">Results</div></div>
            <div class="sw-line {"active" if sa>=4 else ""}"></div>
            <div class="sw-step {"active" if sa>=4 else ""}"><div class="sw-num">4</div><div class="sw-label">Report</div></div>
        </div>''', unsafe_allow_html=True)

        # AUTH STEP 1
        if sa == 1:
            st.markdown('<div class="step-header"><div class="sh-num">01</div><div><div class="sh-title">Upload Evidence</div><div class="sh-desc">Upload any image for tampering detection.</div></div></div>', unsafe_allow_html=True)

            aup = st.file_uploader("Upload file", type=["jpg","jpeg","png"], key="au_up", label_visibility="collapsed")
            if aup is not None:
                fb = np.asarray(bytearray(aup.read()), dtype=np.uint8)
                ai = cv2.imdecode(fb, cv2.IMREAD_COLOR)
                if ai is not None:
                    st.session_state.au_orig = ai
                    st.session_state.au_last = aup.name
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(aup.name)[1])
                    tf.write(fb)
                    st.session_state.au_ip = tf.name
                    st.image(cv2.cvtColor(ai, cv2.COLOR_BGR2RGB), use_container_width=True)

                if st.button("Continue to Analysis  →", type="primary", key="au_next1", use_container_width=True):
                    st.session_state.au_step = 2
                    st.rerun()

        # AUTH STEP 2
        elif sa == 2:
            st.markdown('<div class="step-header"><div class="sh-num">02</div><div><div class="sh-title">Run Analysis</div><div class="sh-desc">Click below to start forensic examination.</div></div></div>', unsafe_allow_html=True)

            if st.session_state.au_orig is not None:
                st.image(cv2.cvtColor(st.session_state.au_orig, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Uploaded evidence")

            if st.button("🔍  Run Forensic Analysis", type="primary", key="au_run", use_container_width=True):
                ip = st.session_state.au_ip
                if not ip:
                    st.error("No image loaded. Go back and upload.")
                else:
                    t0 = time.time()
                    with st.status("Running analysis...", expanded=True) as s:
                        s.write("📊 Computing ELA...")
                        s.write("🔍 Scanning metadata...")
                        s.write("~ Analyzing noise...")
                        res = run_full_authenticity_check(ip)
                        res["time"] = round(time.time() - t0, 1)
                        st.session_state.au_res = res
                        st.session_state.au_fn = st.session_state.au_last
                        s.update(label=f"Complete — {res['time']}s", state="complete")
                        time.sleep(0.5)
                        st.session_state.au_step = 3
                        st.rerun()

            if st.button("←  Back", key="au_back2"):
                st.session_state.au_step = 1
                st.rerun()

        # AUTH STEP 3
        elif sa == 3:
            st.markdown('<div class="step-header"><div class="sh-num">03</div><div><div class="sh-title">Examination Results</div><div class="sh-desc">Forensic findings.</div></div></div>', unsafe_allow_html=True)

            r = st.session_state.au_res
            if r:
                c = risk_color(r["overall_risk"])
                p = r["overall_risk"] * 100
                conf = "HIGH" if p>80 or p<20 else "MODERATE"

                st.markdown(f'''
                <div class="verdict-banner" style="border-color:{c}">
                    <div class="vb-row">
                        <span class="vb-label" style="color:{c}">{r["verdict"]}</span>
                        <span class="vb-time"><span class="vb-time-icon">⏱</span> {r.get("time","?")}s</span>
                    </div>
                    <div class="vb-prob">{p:.1f}% risk score</div>
                    <div class="vb-bar-wrap"><div class="vb-bar" style="width:{p}%;background:{c}"></div></div>
                    <div class="vb-meta">
                        <span class="vb-tag">Confidence: <b>{conf}</b></span>
                        <span class="vb-tag">Time: <b>{r.get('time','?')}s</b></span>
                    </div>
                </div>''', unsafe_allow_html=True)

                m1,m2,m3 = st.columns(3)
                for col,key,label,w in [(m1,"ela","ELA Signal",.45),(m2,"metadata","Metadata",.25),(m3,"noise","Noise",.30)]:
                    v=r[key]["score"]*100; mc=risk_color(r[key]["score"])
                    col.markdown(f'''
                    <div class="metric-card">
                        <div class="mc-top"><div class="mc-label">{label}</div><div class="mc-weight">{int(w*100)}%</div></div>
                        <div class="mc-value" style="color:{mc}">{v:.1f}%</div>
                        <div class="mc-bar"><div class="mc-fill" style="width:{v}%;background:{mc}"></div></div>
                        <div class="mc-assess">{_assess(r[key]["score"])}</div>
                    </div>''', unsafe_allow_html=True)

                st.markdown('<div class="img-card-title" style="margin-top:1.5rem"><span class="ict-dot"></span>ELA Visualization</div>', unsafe_allow_html=True)
                if "image" in r.get("ela",{}):
                    st.image(cv2.cvtColor(r["ela"]["image"], cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown('<div class="img-card-title" style="margin-top:1.5rem"><span class="ict-dot"></span>Metadata</div>', unsafe_allow_html=True)
                for f in r["metadata"]["findings"]: st.write(f"• {f}")
                if r["metadata"].get("exif"):
                    with st.expander("Raw EXIF"): st.json(r["metadata"]["exif"])

                c_b,c_n = st.columns([1,2])
                with c_b:
                    if st.button("←  Re-analyze", key="au_back3"):
                        st.session_state.au_step = 1
                        st.session_state.au_res = None
                        st.rerun()
                with c_n:
                    if st.button("Download Report  →", type="primary", key="au_next3", use_container_width=True):
                        st.session_state.au_step = 4
                        st.rerun()

        # AUTH STEP 4
        elif sa == 4:
            st.markdown('<div class="step-header"><div class="sh-num">04</div><div><div class="sh-title">Forensic Report</div><div class="sh-desc">Download your certified report.</div></div></div>', unsafe_allow_html=True)

            st.markdown('''
            <div class="pdf-hero">
                <div class="pdf-hero-icon">📄</div>
                <div class="pdf-hero-text">
                    <div class="pdf-hero-title">Forensic Examination Report</div>
                    <div class="pdf-hero-desc">Original image, ELA, metadata, metrics & certification.</div>
                </div>
            </div>
            <div class="report-features">
                <div class="rf-item"><span class="rf-check">✓</span> Original image</div>
                <div class="rf-item"><span class="rf-check">✓</span> ELA visualization</div>
                <div class="rf-item"><span class="rf-check">✓</span> Metrics & metadata</div>
                <div class="rf-item"><span class="rf-check">✓</span> Methodology</div>
                <div class="rf-item"><span class="rf-check">✓</span> Certification</div>
                <div class="rf-item"><span class="rf-check">✓</span> CONFIDENTIAL watermark</div>
            </div>''', unsafe_allow_html=True)

            r = st.session_state.au_res
            if r:
                orig = st.session_state.au_orig
                imgs = {"original": orig, "ela": r.get("ela",{}).get("image")}
                op = get_report_temp_path("auth_report.pdf")
                build_report(op, st.session_state.au_fn, "authenticity", r, images=imgs)
                with open(op,"rb") as f:
                    st.download_button("⬇️  Download Forensic Report (PDF)", f,
                        file_name=f"forensic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", key="au_dl", use_container_width=True)

            if st.button("←  Analyze Another File", key="au_back4"):
                st.session_state.au_step = 1
                for k in ["au_res","au_orig","au_last","au_ip","au_fn"]:
                    st.session_state[k] = None
                st.rerun()