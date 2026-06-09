import os
# Must be set BEFORE any ultralytics/cv2 import
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import numpy as np
import io
import csv
import time
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelVision AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
:root{--navy:#060d1a;--navy2:#0a1628;--navy3:#0f1f3a;--panel:#0d1b2e;--green:#39ff14;--green2:#1aff6e;--green-dim:#1a5c0a;--amber:#ffb300;--border:#1c3a5e;--text:#c8daf0;--text-dim:#5a7a9a;--scan:rgba(0,255,60,0.03)}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--navy)!important;color:var(--text)!important;font-family:'Exo 2',sans-serif!important}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,var(--scan) 2px,var(--scan) 4px);pointer-events:none;z-index:9999}
[data-testid="stSidebar"]{background:var(--navy2)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
h1,h2,h3{font-family:'Orbitron',monospace!important;letter-spacing:.05em}
h1{color:var(--green)!important;text-shadow:0 0 20px rgba(57,255,20,.4)}
h2{color:var(--green2)!important;font-size:1.1rem!important}
h3{color:var(--amber)!important;font-size:.95rem!important}
[data-testid="stMetric"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-top:2px solid var(--green)!important;border-radius:4px!important;padding:12px 16px!important}
[data-testid="stMetricLabel"]{font-family:'Share Tech Mono',monospace!important;font-size:.7rem!important;color:var(--text-dim)!important;letter-spacing:.12em!important;text-transform:uppercase!important}
[data-testid="stMetricValue"]{font-family:'Orbitron',monospace!important;font-size:1.6rem!important;color:var(--green)!important;text-shadow:0 0 8px rgba(57,255,20,.3)!important}
.stButton>button{background:transparent!important;border:1px solid var(--green)!important;color:var(--green)!important;font-family:'Orbitron',monospace!important;font-size:.7rem!important;letter-spacing:.1em!important;border-radius:2px!important;transition:all .2s!important}
.stButton>button:hover{background:var(--green)!important;color:var(--navy)!important;box-shadow:0 0 16px rgba(57,255,20,.5)!important}
[data-testid="stDownloadButton"] button{background:transparent!important;border:1px solid var(--amber)!important;color:var(--amber)!important;font-family:'Orbitron',monospace!important;font-size:.65rem!important;letter-spacing:.1em!important;border-radius:2px!important}
[data-testid="stDownloadButton"] button:hover{background:var(--amber)!important;color:var(--navy)!important;box-shadow:0 0 14px rgba(255,179,0,.4)!important}
.stTabs [data-baseweb="tab-list"]{background:var(--navy2)!important;border-bottom:1px solid var(--border)!important;gap:0!important}
.stTabs [data-baseweb="tab"]{font-family:'Orbitron',monospace!important;font-size:.65rem!important;letter-spacing:.08em!important;color:var(--text-dim)!important;border-radius:0!important;padding:10px 20px!important;border-bottom:2px solid transparent!important}
.stTabs [aria-selected="true"]{color:var(--green)!important;border-bottom:2px solid var(--green)!important;background:var(--panel)!important}
[data-testid="stFileUploadDropzone"]{background:var(--panel)!important;border:1px dashed var(--border)!important;border-radius:4px!important}
[data-testid="stFileUploadDropzone"]:hover{border-color:var(--green)!important}
.stAlert{background:var(--panel)!important;border-left:3px solid var(--green)!important;font-family:'Share Tech Mono',monospace!important;font-size:.8rem!important}
hr{border-color:var(--border)!important;margin:1.5rem 0!important}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--navy)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.header-banner{background:linear-gradient(135deg,var(--navy2) 0%,var(--navy3) 50%,#061428 100%);border:1px solid var(--border);border-top:3px solid var(--green);border-radius:4px;padding:24px 32px;margin-bottom:24px;position:relative;overflow:hidden}
.header-banner::after{content:"";position:absolute;top:0;right:0;width:200px;height:200px;background:radial-gradient(circle,rgba(57,255,20,.06) 0%,transparent 70%);pointer-events:none}
.header-title{font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;color:#39ff14;text-shadow:0 0 24px rgba(57,255,20,.5);letter-spacing:.08em;margin:0}
.header-sub{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:#5a7a9a;letter-spacing:.18em;text-transform:uppercase;margin-top:6px}
.status-dot{display:inline-block;width:8px;height:8px;background:var(--green);border-radius:50%;box-shadow:0 0 8px var(--green);animation:pulse 1.4s ease-in-out infinite;margin-right:8px}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.85)}}
.badge{display:inline-block;background:var(--green-dim);color:var(--green);font-family:'Share Tech Mono',monospace;font-size:.6rem;letter-spacing:.15em;padding:2px 8px;border:1px solid var(--green);border-radius:2px;margin-right:6px}
.section-label{font-family:'Share Tech Mono',monospace;font-size:.65rem;color:var(--text-dim);letter-spacing:.2em;text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:6px;margin-bottom:14px}
.perf-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-family:'Share Tech Mono',monospace;font-size:.72rem}
.perf-val{color:var(--green);font-weight:600}
.class-chip{display:inline-block;background:var(--navy3);border:1px solid var(--border);color:var(--text);font-family:'Share Tech Mono',monospace;font-size:.68rem;padding:3px 10px;border-radius:2px;margin:3px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL — loaded once, cached
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Ultralytics internally imports cv2 which on Streamlit Cloud raises
    'libGL.so.1: cannot open shared object file' unless we patch cv2.imshow
    before the import happens. We use the headless trick: set the env var
    and then monkey-patch so the import succeeds silently.
    """
    try:
        # Prevent cv2 from trying to open a display
        import sys

        # Attempt normal import first
        try:
            import cv2
        except ImportError:
            # cv2 not available — create a minimal stub so ultralytics doesn't crash
            import types
            cv2_stub = types.ModuleType("cv2")
            cv2_stub.__version__ = "4.9.0"
            cv2_stub.imencode = lambda *a, **k: (True, np.array([]))
            cv2_stub.imdecode = lambda *a, **k: np.zeros((100,100,3), np.uint8)
            cv2_stub.cvtColor = lambda img, *a, **k: img
            cv2_stub.COLOR_BGR2RGB = 4
            cv2_stub.COLOR_RGB2BGR = 4
            cv2_stub.IMREAD_COLOR = 1
            sys.modules["cv2"] = cv2_stub

        from ultralytics import YOLO
        model = YOLO("best.pt")
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# CLASS CONFIG
# NOTE: ultralytics remaps class IDs to 0,1,2 during training.
# Original dataset: 0=airplane, 13=ship, 18=vehicle
# After training with 3 classes: 0=airplane, 1=ship, 2=vehicle
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "airplane": (57, 255, 20),    # neon green
    "ship":     (0, 200, 255),    # cyan
    "vehicle":  (255, 179, 0),    # amber
}
CLASS_EMOJI = {"airplane": "✈️", "ship": "🚢", "vehicle": "🚗"}

# Remapped IDs from training (ultralytics always uses 0-indexed for filtered datasets)
CLASS_IDS = {0: "airplane", 1: "ship", 2: "vehicle"}


# ─────────────────────────────────────────────
# PIL-ONLY OBB DRAWING — no cv2 calls in app code
# ─────────────────────────────────────────────
def draw_obb_pil(image_pil, detections_raw):
    img = image_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 13)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    for det in detections_raw:
        cls_name = det["class"]
        conf     = det["confidence"]
        pts      = det["pts"]
        r, g, b  = CLASS_COLORS.get(cls_name, (255, 255, 255))

        draw.polygon(pts, fill=(r, g, b, 25), outline=(r, g, b, 230))
        # Draw extra passes for thicker border
        for _ in range(2):
            draw.polygon(pts, outline=(r, g, b, 200))

        # Label chip
        lx, ly = int(pts[0][0]), int(pts[0][1])
        ly = max(ly, 18)
        label = f" {cls_name.upper()} {conf:.2f} "
        bbox  = draw.textbbox((lx, ly - 18), label, font=font)
        draw.rectangle(bbox, fill=(r, g, b, 220))
        draw.text((lx, ly - 18), label, fill=(0, 0, 0, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, image_pil, conf_threshold):
    img_np  = np.array(image_pil.convert("RGB"))
    results = model.predict(img_np, conf=conf_threshold, verbose=False)
    result  = results[0]

    detections_raw = []
    detections     = []

    if result.obb is not None and len(result.obb) > 0:
        for box in result.obb:
            cls_id   = int(box.cls[0].item())
            conf     = float(box.conf[0].item())
            cls_name = CLASS_IDS.get(cls_id, f"class_{cls_id}")
            corners  = box.xyxyxyxy[0].cpu().numpy().reshape(-1, 2)
            pts      = [(float(x), float(y)) for x, y in corners]

            detections_raw.append({"class": cls_name, "confidence": conf, "pts": pts})
            detections.append({"class": cls_name, "confidence": conf})

    annotated = draw_obb_pil(image_pil, detections_raw)
    return annotated, detections


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def make_bar_chart(counts):
    import plotly.graph_objects as go
    labels = [k.capitalize() for k in counts]
    values = list(counts.values())
    colors = [f"rgb{CLASS_COLORS.get(k,(100,200,100))}" for k in counts]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        marker_line_color="rgba(255,255,255,0.1)", marker_line_width=1,
        text=values, textposition="outside",
        textfont=dict(family="Share Tech Mono", size=12, color="#c8daf0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,46,0.6)",
        font=dict(family="Share Tech Mono", color="#5a7a9a"),
        xaxis=dict(gridcolor="rgba(28,58,94,0.5)", zeroline=False),
        yaxis=dict(gridcolor="rgba(28,58,94,0.5)", zeroline=False, title="Count"),
        margin=dict(l=20,r=20,t=20,b=20), height=260,
    )
    return fig


def make_conf_hist(detections):
    import plotly.graph_objects as go
    if not detections:
        return None
    classes = list({d["class"] for d in detections})
    fig = go.Figure()
    for cls in classes:
        vals = [d["confidence"] for d in detections if d["class"] == cls]
        rgb  = f"rgb{CLASS_COLORS.get(cls,(100,200,100))}"
        fig.add_trace(go.Histogram(
            x=vals, name=cls.capitalize(), nbinsx=10,
            marker_color=rgb, opacity=0.8,
        ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,46,0.6)",
        font=dict(family="Share Tech Mono", color="#5a7a9a"),
        xaxis=dict(title="Confidence", gridcolor="rgba(28,58,94,0.5)", range=[0,1]),
        yaxis=dict(title="Count", gridcolor="rgba(28,58,94,0.5)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(28,58,94,0.5)", borderwidth=1),
        margin=dict(l=20,r=20,t=20,b=20), height=260,
    )
    return fig


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_csv(detections):
    out = io.StringIO()
    w   = csv.writer(out)
    w.writerow(["#","Class","Confidence","Status"])
    for i, d in enumerate(detections, 1):
        status = "CONFIRMED" if d["confidence"] >= 0.5 else "PROBABLE"
        w.writerow([i, d["class"].capitalize(), f"{d['confidence']:.4f}", status])
    w.writerow([])
    w.writerow(["SUMMARY","","",""])
    w.writerow(["Total Objects", len(detections),"",""])
    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    for cls, cnt in counts.items():
        w.writerow([f"  {cls.capitalize()}", cnt,"",""])
    if detections:
        avg = sum(d["confidence"] for d in detections) / len(detections)
        w.writerow(["Average Confidence", f"{avg:.4f}","",""])
    return out.getvalue().encode()


def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
  <p class="header-title">🛰️ SENTINELVISION AI</p>
  <p class="header-sub">
    <span class="status-dot"></span>SYSTEM ONLINE &nbsp;│&nbsp;
    Defence Satellite Object Detection &nbsp;│&nbsp;
    <span class="badge">YOLO11s-OBB</span>
    <span class="badge">DIOR-R</span>
    <span class="badge">mAP50 92.9%</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown('<p class="section-label">🔧 Detection Controls</p>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90,
        value=0.25, step=0.01,
        help="Filter out detections below this confidence score",
    )
    st.caption(f"Active threshold: `{conf_threshold:.2f}`")

    st.markdown("---")
    st.markdown('<p class="section-label">📊 Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="perf-row"><span>Architecture</span><span class="perf-val">YOLO11s-OBB</span></div>
    <div class="perf-row"><span>Dataset</span><span class="perf-val">DIOR-R (3-class)</span></div>
    <div class="perf-row"><span>Precision</span><span class="perf-val">92.9%</span></div>
    <div class="perf-row"><span>Recall</span><span class="perf-val">87.7%</span></div>
    <div class="perf-row"><span>mAP@50</span><span class="perf-val">92.9%</span></div>
    <div class="perf-row"><span>mAP@50-95</span><span class="perf-val">77.1%</span></div>
    <div class="perf-row"><span>Inference</span><span class="perf-val">7.8 ms/img</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">🎯 Target Classes</p>', unsafe_allow_html=True)
    for cls, emoji in CLASS_EMOJI.items():
        st.markdown(f'<span class="class-chip">{emoji} {cls.capitalize()}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📡 Class Performance</p>', unsafe_allow_html=True)
    for cls, (prec, rec, map50) in {
        "Airplane": ("97.3%","97.3%","98.4%"),
        "Ship":     ("95.0%","96.3%","98.2%"),
        "Vehicle":  ("86.4%","69.6%","82.1%"),
    }.items():
        st.markdown(f"""
        <div style="margin-bottom:10px">
          <div style="font-family:'Orbitron',monospace;font-size:.65rem;color:#ffb300;margin-bottom:4px">{cls.upper()}</div>
          <div class="perf-row" style="font-size:.65rem"><span>Precision</span><span class="perf-val">{prec}</span></div>
          <div class="perf-row" style="font-size:.65rem"><span>Recall</span><span class="perf-val">{rec}</span></div>
          <div class="perf-row" style="font-size:.65rem"><span>mAP50</span><span class="perf-val">{map50}</span></div>
        </div>""", unsafe_allow_html=True)

# ── LOAD MODEL ──
with st.spinner("🔄 Initialising YOLO11s-OBB model…"):
    model, model_error = load_model()

if model_error:
    st.error(f"""
    ⚠️ **Model failed to load.**

    **Error:** `{model_error}`

    **Checklist:**
    - Ensure `best.pt` is committed to your repo root (same folder as `app.py`)
    - File must be named exactly `best.pt` (lowercase)
    - If file is >100 MB, GitHub will reject it — use Git LFS or host on Hugging Face Hub
    - Check the Streamlit Cloud logs for the full traceback
    """)
    st.stop()

st.success("✅ YOLO11s-OBB loaded — system ready for detection")

# ── TABS ──
tab_detect, tab_info, tab_results = st.tabs([
    "🛰️  DETECTION",
    "📋  PROJECT INFO",
    "📊  TRAINING RESULTS",
])


# ════════════════════════════════════════════
# TAB 1 — DETECTION
# ════════════════════════════════════════════
with tab_detect:
    st.markdown('<p class="section-label">📁 Upload Satellite Imagery</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload satellite image",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
        help="Supported formats: JPG, JPEG, PNG. Recommended: 640×640 or larger.",
    )

    if uploaded is None:
        # ── EMPTY STATE ──
        st.markdown("""
        <div style="text-align:center;padding:56px 24px;
             background:rgba(13,27,46,0.6);
             border:1px dashed rgba(28,58,94,0.8);
             border-radius:4px;margin-top:12px">
          <p style="font-family:'Orbitron',monospace;font-size:1.4rem;
             color:#1c3a5e;margin:0;letter-spacing:.1em">📡</p>
          <p style="font-family:'Orbitron',monospace;font-size:1rem;
             color:#1c3a5e;margin:8px 0 0;letter-spacing:.1em">
            AWAITING SATELLITE FEED</p>
          <p style="font-family:'Share Tech Mono',monospace;font-size:.72rem;
             color:#5a7a9a;letter-spacing:.15em;margin-top:10px">
            UPLOAD A SATELLITE IMAGE TO INITIATE OBJECT DETECTION</p>
          <p style="font-family:'Share Tech Mono',monospace;font-size:.65rem;
             color:#2a4a6a;margin-top:16px">
            Detects: ✈️ Airplanes &nbsp;|&nbsp; 🚢 Ships &nbsp;|&nbsp; 🚗 Vehicles</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        try:
            image_pil = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"⚠️ Could not open image: `{e}`. Please upload a valid JPG or PNG file.")
            st.stop()

        w, h = image_pil.size

        # Warn if image is very small
        if w < 100 or h < 100:
            st.warning("⚠️ Image resolution is very low. Detection quality may be poor. Recommended minimum: 640×640 px.")

        # ── INFERENCE ──
        with st.spinner("⚙️ Running YOLO11s-OBB inference on satellite imagery…"):
            t0 = time.perf_counter()
            try:
                annotated_pil, detections = run_inference(model, image_pil, conf_threshold)
            except Exception as e:
                st.error(f"⚠️ Inference failed: `{e}`")
                st.info("Try lowering the confidence threshold or uploading a different image.")
                st.stop()
            elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── COUNTS ──
        counts  = {"airplane": 0, "ship": 0, "vehicle": 0}
        for d in detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        avg_conf = (sum(d["confidence"] for d in detections) / len(detections)) if detections else 0.0

        # ── DETECTION RESULT BANNER ──
        if len(detections) == 0:
            st.warning(f"""
            🔍 **No objects detected** at confidence threshold `{conf_threshold:.2f}`.

            **Try:**
            - Lowering the confidence threshold using the sidebar slider
            - Uploading a clearer or higher-resolution satellite image
            - Ensuring the image contains airplanes, ships, or vehicles
            """)
        else:
            st.success(f"✅ **{len(detections)} object(s) detected** — {counts['airplane']} airplane(s), {counts['ship']} ship(s), {counts['vehicle']} vehicle(s)")

        # ── METRICS ──
        st.markdown('<p class="section-label">📊 Detection Summary</p>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("TOTAL OBJECTS",  len(detections))
        m2.metric("✈️ AIRPLANES",   counts["airplane"])
        m3.metric("🚢 SHIPS",       counts["ship"])
        m4.metric("🚗 VEHICLES",    counts["vehicle"])
        m5.metric("AVG CONFIDENCE", f"{avg_conf:.2f}" if detections else "N/A")
        m6.metric("INFER TIME",     f"{elapsed_ms:.0f} ms")

        st.markdown("---")

        # ── IMAGE COMPARISON ──
        st.markdown('<p class="section-label">🖼️ Visual Comparison</p>', unsafe_allow_html=True)
        col_o, col_d = st.columns(2)
        with col_o:
            st.markdown("**ORIGINAL IMAGE**")
            st.image(image_pil, width="stretch")
            st.caption(f"Resolution: {w} × {h} px  |  Format: {uploaded.type}  |  Size: {uploaded.size/1024:.1f} KB")
        with col_d:
            st.markdown("**DETECTED OBJECTS**")
            st.image(annotated_pil, width="stretch")
            if detections:
                st.caption(f"{len(detections)} object(s) detected  |  conf ≥ {conf_threshold:.2f}  |  {elapsed_ms:.0f} ms inference")
            else:
                st.caption(f"No detections at conf ≥ {conf_threshold:.2f} — try lowering threshold")

        st.markdown("---")

        if detections:
            # ── ANALYTICS ──
            st.markdown('<p class="section-label">📈 Visual Analytics</p>', unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**OBJECT COUNT BY CLASS**")
                st.plotly_chart(make_bar_chart(counts),
                                width="stretch", config={"displayModeBar": False})
            with ch2:
                st.markdown("**CONFIDENCE SCORE DISTRIBUTION**")
                hist = make_conf_hist(detections)
                if hist:
                    st.plotly_chart(hist, width="stretch", config={"displayModeBar": False})

            st.markdown("---")

            # ── DETECTION LOG TABLE ──
            st.markdown('<p class="section-label">🗂️ Detection Log</p>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame([{
                "ID":         i+1,
                "Class":      d["class"].capitalize(),
                "Emoji":      CLASS_EMOJI.get(d["class"], "?"),
                "Confidence": f"{d['confidence']:.4f}",
                "Status":     "✅ CONFIRMED" if d["confidence"] >= 0.5 else "⚠️ PROBABLE",
            } for i, d in enumerate(detections)])
            st.dataframe(df, width="stretch", hide_index=True)

            # Per-class breakdown
            st.markdown('<p class="section-label" style="margin-top:16px">🔢 Class Breakdown</p>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            for col, (cls, emoji) in zip([b1,b2,b3], CLASS_EMOJI.items()):
                cls_dets = [d for d in detections if d["class"] == cls]
                if cls_dets:
                    avg_c = sum(d["confidence"] for d in cls_dets) / len(cls_dets)
                    max_c = max(d["confidence"] for d in cls_dets)
                    col.markdown(f"""
                    <div style="background:var(--panel);border:1px solid var(--border);
                         border-top:2px solid {'#39ff14' if cls=='airplane' else '#00c8ff' if cls=='ship' else '#ffb300'};
                         border-radius:4px;padding:12px;text-align:center">
                      <div style="font-size:1.5rem">{emoji}</div>
                      <div style="font-family:'Orbitron',monospace;font-size:.7rem;
                           color:#ffb300;margin:4px 0">{cls.upper()}</div>
                      <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#39ff14">
                        {counts[cls]}</div>
                      <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;color:#5a7a9a;margin-top:4px">
                        Avg conf: {avg_c:.2f} &nbsp;|&nbsp; Max: {max_c:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    col.markdown(f"""
                    <div style="background:var(--panel);border:1px solid var(--border);
                         border-radius:4px;padding:12px;text-align:center;opacity:.4">
                      <div style="font-size:1.5rem">{emoji}</div>
                      <div style="font-family:'Orbitron',monospace;font-size:.7rem;color:#5a7a9a;margin:4px 0">
                        {cls.upper()}</div>
                      <div style="font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#5a7a9a">
                        NOT DETECTED</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ── DOWNLOADS ──
        st.markdown('<p class="section-label">⬇️ Export Results</p>', unsafe_allow_html=True)
        dl1, dl2, _ = st.columns([1,1,2])
        with dl1:
            st.download_button(
                label="⬇ DETECTED IMAGE (PNG)",
                data=pil_to_bytes(annotated_pil),
                file_name=f"sentinelvision_detection_{int(time.time())}.png",
                mime="image/png",
                help="Download the annotated satellite image with bounding boxes",
            )
        with dl2:
            st.download_button(
                label="⬇ DETECTION REPORT (CSV)",
                data=build_csv(detections),
                file_name=f"sentinelvision_report_{int(time.time())}.csv",
                mime="text/csv",
                help="Download the full detection log as a CSV file",
            )

        if not detections:
            st.info("💡 No detections to export. Lower the confidence threshold or try a different image.")


# ════════════════════════════════════════════
# TAB 2 — PROJECT INFO
# ════════════════════════════════════════════
with tab_info:
    c1, c2 = st.columns([3,2])

    with c1:
        st.markdown("## PROJECT OVERVIEW")
        st.markdown("""
        **SentinelVision AI** is a professional AI-powered defence surveillance dashboard
        for detecting high-value military objects from satellite imagery using a
        **YOLO11s-OBB** oriented bounding box detection model trained on the
        **DIOR-R** dataset.

        The system is purpose-built for **Intelligence, Surveillance & Reconnaissance (ISR)**
        operations, capable of identifying three defence-relevant object categories
        in real-time at under 13 ms per image.

        Developed as part of an internship at **DRDO SSPL (Solid State Physics Laboratory)**.
        """)
        st.markdown("---")
        st.markdown("### PROJECT WORKFLOW")
        steps = [
            ("1️⃣", "Dataset Acquisition",   "Downloaded DIOR-R (~7 GB, 20 classes, 23,463 images) with rotated bounding box annotations in YOLO OBB format."),
            ("2️⃣", "Local Filtering",        "Filtered the dataset locally to retain only 3 defence-relevant classes — Airplane (ID 0), Ship (ID 13), Vehicle (ID 18) — reducing it to ~3 GB."),
            ("3️⃣", "Cloud Upload",           "Compressed the filtered dataset (DIOR-3class) and uploaded it as a ZIP to Google Drive for persistent cloud storage."),
            ("4️⃣", "Google Colab Training",  "Mounted Google Drive in Colab, unzipped the dataset, and trained YOLO11s-OBB for 50 epochs on an NVIDIA Tesla T4 GPU (~5 hours). Drive mounting avoided repeated 3 GB uploads each session."),
            ("5️⃣", "Model Evaluation",       "Evaluated best.pt on the test split — achieved mAP@50 of 92.9%. Generated confusion matrices, PR curves, F1 curves, and validation batch predictions."),
            ("6️⃣", "GitHub & Deployment",    "Pushed all artefacts (best.pt, app.py, training charts) to GitHub. Connected the repository to Streamlit Community Cloud for one-click deployment."),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:14px;
                 background:rgba(13,27,46,0.5);border:1px solid var(--border);
                 border-left:3px solid #39ff14;border-radius:3px;padding:10px 14px">
              <div style="font-size:1.2rem;flex-shrink:0">{icon}</div>
              <div>
                <div style="font-family:'Orbitron',monospace;font-size:.65rem;
                     color:#ffb300;letter-spacing:.08em;margin-bottom:3px">{title}</div>
                <div style="font-family:'Exo 2',sans-serif;font-size:.8rem;
                     color:#c8daf0;line-height:1.5">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### ARCHITECTURE DETAILS")
        for k, v in {
            "Model":        "YOLO11s-OBB",
            "Framework":    "Ultralytics YOLO + PyTorch",
            "Dataset":      "DIOR-R (3-class filtered)",
            "Classes":      "Airplane · Ship · Vehicle",
            "Epochs":       "50",
            "Image Size":   "640 × 640 px",
            "Batch Size":   "4",
            "GPU":          "NVIDIA Tesla T4",
            "Optimizer":    "AdamW (auto-selected)",
            "Annotation":   "YOLO OBB 8-point polygon",
        }.items():
            st.markdown(f'<div class="perf-row"><span>{k}</span><span class="perf-val">{v}</span></div>',
                        unsafe_allow_html=True)

    with c2:
        st.markdown("### OVERALL METRICS")
        for k, v in {
            "Precision":   "92.9%",
            "Recall":      "87.7%",
            "mAP@50":      "92.9%",
            "mAP@50-95":   "77.1%",
            "Fitness":     "0.771",
        }.items():
            st.markdown(f'<div class="perf-row"><span>{k}</span><span class="perf-val">{v}</span></div>',
                        unsafe_allow_html=True)

        st.markdown("### INFERENCE SPEED")
        for k, v in {
            "Preprocess":  "0.4 ms",
            "Inference":   "7.8 ms",
            "Postprocess": "4.6 ms",
            "Total":       "~12.8 ms / image",
        }.items():
            st.markdown(f'<div class="perf-row"><span>{k}</span><span class="perf-val">{v}</span></div>',
                        unsafe_allow_html=True)

        st.markdown("### CLASS-WISE RESULTS")
        for cls, emoji, (prec, rec, map50) in [
            ("Airplane", "✈️", ("97.3%","97.3%","98.4%")),
            ("Ship",     "🚢", ("95.0%","96.3%","98.2%")),
            ("Vehicle",  "🚗", ("86.4%","69.6%","82.1%")),
        ]:
            st.markdown(f"""
            <div style="margin:8px 0 4px;font-family:'Orbitron',monospace;
                 font-size:.65rem;color:#ffb300">{emoji} {cls.upper()}</div>
            <div class="perf-row" style="font-size:.68rem"><span>Precision</span><span class="perf-val">{prec}</span></div>
            <div class="perf-row" style="font-size:.68rem"><span>Recall</span><span class="perf-val">{rec}</span></div>
            <div class="perf-row" style="font-size:.68rem"><span>mAP50</span><span class="perf-val">{map50}</span></div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### DATASET INFORMATION")
    d1, d2, d3 = st.columns(3)
    d1.metric("Original Classes","20");  d1.metric("Total Images","23,463")
    d2.metric("Filtered Classes","3");   d2.metric("Label Format","YOLO OBB")
    d3.metric("OBB Points","8-point");   d3.metric("Splits","Train / Val / Test")

    st.markdown("---")

    # ── PROJECT WORKFLOW / STORY ──────────────────────────────────────────
    st.markdown("### 🗺️ PROJECT WORKFLOW")
    st.markdown("""
    <div style="font-family:'Exo 2',sans-serif;font-size:.85rem;color:#c8daf0;line-height:1.8">
    The complete end-to-end pipeline from raw data to live deployment, developed during an
    internship at <strong style="color:#39ff14">DRDO — SSPL (Solid State Physics Laboratory)</strong>.
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "#39ff14", "📦 Dataset Acquisition",
         "Started with the full <strong>DIOR-R</strong> dataset — 23,463 satellite images across "
         "20 object classes (~7 GB). DIOR-R uses oriented bounding box (OBB) annotations, "
         "making it ideal for satellite imagery where objects appear at arbitrary rotations."),

        ("02", "#00c8ff", "🔧 Local Dataset Filtering",
         "Filtered the full 20-class dataset down to the 3 defence-relevant classes — "
         "<strong>airplane, ship, vehicle</strong> — locally, reducing the dataset to ~3 GB. "
         "All other class annotations and images containing only non-target objects were removed. "
         "The filtered dataset was packaged as a zip and uploaded to Google Drive."),

        ("03", "#ffb300", "☁️ Google Colab + Drive Setup",
         "Mounted Google Drive inside Google Colab to access the 3 GB filtered dataset "
         "without re-uploading each session. This enabled fast data loading directly from Drive "
         "into the training pipeline, avoiding the slow manual upload bottleneck."),

        ("04", "#39ff14", "🏋️ Model Training — YOLO11s-OBB",
         "Trained <strong>YOLO11s-OBB</strong> (pretrained on COCO) for 50 epochs on an "
         "<strong>NVIDIA Tesla T4 GPU</strong> (Colab). Used AdamW optimizer (auto-selected), "
         "640×640 resolution, batch size 4. Total training time: ~5 hours. "
         "Best checkpoint saved as <code>best.pt</code>."),

        ("05", "#00c8ff", "📊 Evaluation & Testing",
         "Evaluated the trained model on the held-out test set. Analysed precision-recall curves, "
         "confusion matrices, and F1-confidence curves per class. Ran inference on sample "
         "satellite images to visually verify OBB quality and class discrimination."),

        ("06", "#ffb300", "🚀 Deployment — Streamlit Cloud",
         "Pushed <code>best.pt</code>, <code>app.py</code>, training artefacts, and "
         "<code>requirements.txt</code> to GitHub. Connected the repository to "
         "<strong>Streamlit Community Cloud</strong> for free, zero-infrastructure deployment. "
         "App is publicly accessible and runs inference live in the browser."),
    ]

    for num, color, title, body in steps:
        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:14px;
             background:rgba(13,27,46,0.5);border:1px solid var(--border);
             border-left:3px solid {color};border-radius:4px;padding:14px 16px">
          <div style="font-family:'Orbitron',monospace;font-size:1.1rem;
               color:{color};opacity:.5;min-width:32px;padding-top:2px">{num}</div>
          <div>
            <div style="font-family:'Orbitron',monospace;font-size:.72rem;
                 color:{color};letter-spacing:.08em;margin-bottom:5px">{title}</div>
            <div style="font-family:'Exo 2',sans-serif;font-size:.82rem;
                 color:#a0b8d0;line-height:1.6">{body}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── DRDO SSPL BADGE ──────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(13,27,46,0.9),rgba(15,31,58,0.9));
         border:1px solid var(--border);border-top:2px solid #39ff14;
         border-radius:4px;padding:18px 24px;text-align:center">
      <div style="font-family:'Orbitron',monospace;font-size:.65rem;
           color:#5a7a9a;letter-spacing:.2em;margin-bottom:6px">DEVELOPED DURING INTERNSHIP AT</div>
      <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:800;
           color:#39ff14;text-shadow:0 0 16px rgba(57,255,20,.4);letter-spacing:.1em">
        DRDO — SSPL</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.72rem;
           color:#c8daf0;margin-top:4px">
        Defence Research & Development Organisation<br>
        Solid State Physics Laboratory, New Delhi</div>
    </div>

    <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem;
         color:#5a7a9a;text-align:center;letter-spacing:.15em;padding:14px 0 4px">
    SENTINELVISION AI &nbsp;│&nbsp; YOLO11s-OBB &nbsp;│&nbsp;
    DIOR-R DATASET &nbsp;│&nbsp; DRDO SSPL &nbsp;│&nbsp; ULTRALYTICS + STREAMLIT
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — TRAINING RESULTS
# ════════════════════════════════════════════
with tab_results:
    st.markdown('<p class="section-label">📊 Model Training & Evaluation Results</p>',
                unsafe_allow_html=True)
    st.markdown("""
    > Training artefacts generated after 50 epochs on NVIDIA Tesla T4 GPU using the
    > DIOR-R 3-class filtered dataset. All charts are saved from the Ultralytics training run.
    """)

    # ── helper to load image from repo with a friendly fallback ──
    def show_result_img(path: str, caption: str, key: str):
        """Try to load a local file; show a placeholder panel if missing."""
        from pathlib import Path
        p = Path(path)
        if p.exists():
            img = Image.open(p)
            st.image(img, caption=caption, width="stretch")
        else:
            st.markdown(f"""
            <div style="background:rgba(13,27,46,0.7);border:1px dashed rgba(28,58,94,0.8);
                 border-radius:4px;padding:28px;text-align:center">
              <p style="font-family:'Share Tech Mono',monospace;font-size:.65rem;
                 color:#2a4a6a;letter-spacing:.12em;margin:0">
                {caption}<br><span style="color:#1c3a5e">{path}</span><br>
                <span style="color:#1a3a5a">FILE NOT FOUND IN REPO</span>
              </p>
            </div>""", unsafe_allow_html=True)

    # ── SECTION 1: Training Curves ──
    st.markdown("### 📈 Training Curves")
    st.caption("Overall loss and metric progression across 50 epochs.")
    show_result_img("results.png", "Training Results — Loss & Metrics over 50 Epochs", "results")

    st.markdown("---")

    # ── SECTION 2: Precision / Recall / F1 Curves ──
    st.markdown("### 🎯 Precision · Recall · F1 Curves")
    col_pr1, col_pr2 = st.columns(2)
    with col_pr1:
        show_result_img("BoxPR_curve.png",  "Precision-Recall Curve", "pr")
        show_result_img("BoxP_curve.png",   "Precision vs Confidence", "pc")
    with col_pr2:
        show_result_img("BoxR_curve.png",   "Recall vs Confidence", "rc")
        show_result_img("BoxF1_curve.png",  "F1 Score vs Confidence", "f1")

    st.markdown("---")

    # ── SECTION 3: Confusion Matrices ──
    st.markdown("### 🔢 Confusion Matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        show_result_img("confusion_matrix.png",
                        "Confusion Matrix (Raw Counts)", "cm_raw")
    with cm2:
        show_result_img("confusion_matrix_normalized.png",
                        "Confusion Matrix (Normalised)", "cm_norm")

    # Explain the confusion matrix for context
    st.markdown("""
    <div style="background:rgba(13,27,46,0.5);border:1px solid var(--border);
         border-left:3px solid #ffb300;border-radius:3px;
         padding:12px 16px;margin-top:8px">
      <div style="font-family:'Orbitron',monospace;font-size:.65rem;
           color:#ffb300;margin-bottom:6px">ℹ️ READING THE CONFUSION MATRIX</div>
      <div style="font-family:'Exo 2',sans-serif;font-size:.8rem;color:#c8daf0;line-height:1.6">
        Rows = actual class, Columns = predicted class.
        The diagonal (top-left to bottom-right) represents correct detections.
        Off-diagonal cells are misclassifications — e.g. a vehicle predicted as background.
        The normalised matrix shows rates (0–1) making cross-class comparison easier.
        Vehicle shows lower recall (69.6%) visible as more off-diagonal mass — 
        expected since small vehicles at satellite altitude are harder to distinguish.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── SECTION 4: Validation Predictions ──
    st.markdown("### 🖼️ Validation Batch Predictions")
    st.caption("Sample predictions on the validation set — green boxes = ground truth, coloured = model predictions.")
    vb1, vb2, vb3 = st.columns(3)
    for col, fname, label in [
        (vb1, "val_batch0_pred.jpg", "Val Batch 0"),
        (vb2, "val_batch1_pred.jpg", "Val Batch 1"),
        (vb3, "val_batch2_pred.jpg", "Val Batch 2"),
    ]:
        with col:
            show_result_img(fname, label, fname)

    st.markdown("---")

    # ── SECTION 5: Final metrics recap ──
    st.markdown("### 🏆 Final Model Performance")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Overall mAP@50",    "92.9%")
    r2.metric("Overall mAP@50-95", "77.1%")
    r3.metric("Precision",         "92.9%")
    r4.metric("Recall",            "87.7%")

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.65rem;
         color:#5a7a9a;text-align:center;letter-spacing:.15em;padding:12px 0">
    SENTINELVISION AI &nbsp;│&nbsp; YOLO11s-OBB &nbsp;│&nbsp;
    DIOR-R DATASET &nbsp;│&nbsp; DRDO SSPL &nbsp;│&nbsp; ULTRALYTICS + STREAMLIT
    </div>""", unsafe_allow_html=True)
