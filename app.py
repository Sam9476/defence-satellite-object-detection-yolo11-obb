import streamlit as st
import numpy as np
import cv2
import io
import csv
import time
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import base64

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelVision AI — Defence Satellite Object Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Military Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

/* ── Root Variables ── */
:root {
    --navy:       #060d1a;
    --navy2:      #0a1628;
    --navy3:      #0f1f3a;
    --panel:      #0d1b2e;
    --green:      #39ff14;
    --green2:     #1aff6e;
    --green-dim:  #1a5c0a;
    --amber:      #ffb300;
    --red:        #ff3a3a;
    --border:     #1c3a5e;
    --text:       #c8daf0;
    --text-dim:   #5a7a9a;
    --scan-line:  rgba(0, 255, 60, 0.04);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* Scanline overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        var(--scan-line) 2px,
        var(--scan-line) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.05em;
}
h1 { color: var(--green) !important; text-shadow: 0 0 20px rgba(57,255,20,0.4); }
h2 { color: var(--green2) !important; font-size: 1.1rem !important; }
h3 { color: var(--amber) !important; font-size: 0.95rem !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--green) !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem !important;
    color: var(--green) !important;
    text-shadow: 0 0 8px rgba(57,255,20,0.3) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--green) !important;
    color: var(--navy) !important;
    box-shadow: 0 0 16px rgba(57,255,20,0.5) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    accent-color: var(--green) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--panel) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--green) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--navy2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-dim) !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
    background: var(--panel) !important;
}

/* ── Charts ── */
.js-plotly-plot, .vega-embed {
    background: transparent !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Caption / small text ── */
.stCaption, small, [data-testid="stCaptionContainer"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-dim) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Info / success / warning boxes ── */
.stAlert {
    background: var(--panel) !important;
    border-left: 3px solid var(--green) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Code blocks ── */
code {
    background: var(--navy3) !important;
    color: var(--green2) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--green-dim); }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--amber) !important;
    color: var(--amber) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--amber) !important;
    color: var(--navy) !important;
    box-shadow: 0 0 14px rgba(255,179,0,0.4) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, var(--navy2) 0%, var(--navy3) 50%, #061428 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--green);
    border-radius: 4px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-banner::after {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(57,255,20,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
    color: #39ff14;
    text-shadow: 0 0 24px rgba(57,255,20,0.5);
    letter-spacing: 0.08em;
    margin: 0;
}
.header-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #5a7a9a;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 6px;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--green);
    animation: pulse 1.4s ease-in-out infinite;
    margin-right: 8px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}
.badge {
    display: inline-block;
    background: var(--green-dim);
    color: var(--green);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    padding: 2px 8px;
    border: 1px solid var(--green);
    border-radius: 2px;
    margin-right: 6px;
}
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 14px;
}
.class-chip {
    display: inline-block;
    background: var(--navy3);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 2px;
    margin: 3px;
}
.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
}
.perf-val { color: var(--green); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAZY IMPORT HEAVY DEPS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("best.pt")
        return model, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
# COLOUR MAP FOR CLASSES
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "airplane": (57, 255, 20),    # neon green
    "ship":     (0, 200, 255),    # cyan
    "vehicle":  (255, 179, 0),    # amber
}
CLASS_EMOJI = {"airplane": "✈️", "ship": "🚢", "vehicle": "🚗"}
CLASS_IDS   = {0: "airplane", 13: "ship", 18: "vehicle"}

# ─────────────────────────────────────────────
# DETECTION LOGIC
# ─────────────────────────────────────────────
def run_inference(model, image_pil, conf_threshold):
    """Run YOLO OBB inference and return annotated image + detections list."""
    img_np = np.array(image_pil.convert("RGB"))
    results = model.predict(img_np, conf=conf_threshold, verbose=False)
    result  = results[0]

    detections = []
    annotated  = img_np.copy()

    if result.obb is not None and len(result.obb) > 0:
        for box in result.obb:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            cls_name = CLASS_IDS.get(cls_id, f"class_{cls_id}")
            color  = CLASS_COLORS.get(cls_name, (255, 255, 255))

            # Oriented bounding box corners (xyxyxyxy)
            pts = box.xyxyxyxy[0].cpu().numpy().reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Label
            x, y = pts[0][0]
            label = f"{cls_name.upper()} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x, y - lh - 6), (x + lw + 4, y), color, -1)
            cv2.putText(annotated, label, (x + 2, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            detections.append({"class": cls_name, "confidence": conf})

    return Image.fromarray(annotated), detections

# ─────────────────────────────────────────────
# HELPER — PLOTLY DARK CHART
# ─────────────────────────────────────────────
def make_bar_chart(counts: dict):
    import plotly.graph_objects as go
    labels = [k.capitalize() for k in counts]
    values = list(counts.values())
    colors = [f"rgb{CLASS_COLORS.get(k,(100,200,100))}" for k in counts]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.15)",
        marker_line_width=1,
        text=values, textposition="outside",
        textfont=dict(family="Share Tech Mono", size=11, color="#c8daf0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.6)",
        font=dict(family="Share Tech Mono", color="#5a7a9a"),
        xaxis=dict(gridcolor="rgba(28,58,94,0.5)", zeroline=False),
        yaxis=dict(gridcolor="rgba(28,58,94,0.5)", zeroline=False, title="Count"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=260,
    )
    return fig


def make_conf_hist(detections):
    import plotly.graph_objects as go
    if not detections:
        return None
    confs = [d["confidence"] for d in detections]
    classes = [d["class"] for d in detections]
    unique = list(set(classes))

    fig = go.Figure()
    for cls in unique:
        c_vals = [d["confidence"] for d in detections if d["class"] == cls]
        rgb = f"rgb{CLASS_COLORS.get(cls,(100,200,100))}"
        fig.add_trace(go.Histogram(
            x=c_vals, name=cls.capitalize(), nbinsx=10,
            marker_color=rgb, opacity=0.8,
        ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.6)",
        font=dict(family="Share Tech Mono", color="#5a7a9a"),
        xaxis=dict(title="Confidence", gridcolor="rgba(28,58,94,0.5)", range=[0, 1]),
        yaxis=dict(title="Count", gridcolor="rgba(28,58,94,0.5)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(28,58,94,0.5)", borderwidth=1),
        margin=dict(l=20, r=20, t=20, b=20),
        height=260,
    )
    return fig

# ─────────────────────────────────────────────
# DETECTION SUMMARY CSV
# ─────────────────────────────────────────────
def build_csv(detections):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Class", "Confidence"])
    for i, d in enumerate(detections, 1):
        writer.writerow([i, d["class"].capitalize(), f"{d['confidence']:.4f}"])
    writer.writerow([])
    writer.writerow(["Total Objects", len(detections)])
    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    for cls, cnt in counts.items():
        writer.writerow([f"  {cls.capitalize()}", cnt])
    if detections:
        avg_conf = sum(d["confidence"] for d in detections) / len(detections)
        writer.writerow(["Average Confidence", f"{avg_conf:.4f}"])
    return output.getvalue().encode()


def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── HEADER ──
st.markdown("""
<div class="header-banner">
  <p class="header-title">🛰️ SENTINELVISION AI</p>
  <p class="header-sub">
    <span class="status-dot"></span>SYSTEM ONLINE &nbsp;│&nbsp;
    Defence Satellite Object Detection System &nbsp;│&nbsp;
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
        help="Filter detections below this confidence score",
    )
    st.caption(f"Active threshold: `{conf_threshold:.2f}`")

    st.markdown("---")
    st.markdown('<p class="section-label">📊 Model Information</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="perf-row"><span>Architecture</span><span class="perf-val">YOLO11s-OBB</span></div>
    <div class="perf-row"><span>Dataset</span><span class="perf-val">DIOR-R</span></div>
    <div class="perf-row"><span>Precision</span><span class="perf-val">92.9%</span></div>
    <div class="perf-row"><span>Recall</span><span class="perf-val">87.7%</span></div>
    <div class="perf-row"><span>mAP@50</span><span class="perf-val">92.9%</span></div>
    <div class="perf-row"><span>mAP@50-95</span><span class="perf-val">77.1%</span></div>
    <div class="perf-row"><span>Inference</span><span class="perf-val">7.8 ms</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">🎯 Target Classes</p>', unsafe_allow_html=True)
    for cls, emoji in CLASS_EMOJI.items():
        st.markdown(f'<span class="class-chip">{emoji} {cls.capitalize()}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📡 Class Performance</p>', unsafe_allow_html=True)
    class_perf = {
        "Airplane": ("97.3%", "97.3%", "98.4%"),
        "Ship":     ("95.0%", "96.3%", "98.2%"),
        "Vehicle":  ("86.4%", "69.6%", "82.1%"),
    }
    for cls, (prec, rec, map50) in class_perf.items():
        st.markdown(f"""
        <div style="margin-bottom:10px">
          <div style="font-family:'Orbitron',monospace;font-size:0.65rem;color:#ffb300;margin-bottom:4px">{cls.upper()}</div>
          <div class="perf-row" style="font-size:0.65rem"><span>P</span><span class="perf-val">{prec}</span></div>
          <div class="perf-row" style="font-size:0.65rem"><span>R</span><span class="perf-val">{rec}</span></div>
          <div class="perf-row" style="font-size:0.65rem"><span>mAP50</span><span class="perf-val">{map50}</span></div>
        </div>
        """, unsafe_allow_html=True)

# ── LOAD MODEL ──
with st.spinner("🔄 Initialising YOLO11s-OBB model..."):
    model, model_error = load_model()

if model_error:
    st.error(f"⚠️ Model load failed: {model_error}")
    st.info("Ensure `best.pt` is in the same directory as `app.py`.")
    st.stop()

st.success("✅ Model loaded — YOLO11s-OBB ready for inference")

# ── TABS ──
tab_detect, tab_info = st.tabs(["🛰️  DETECTION", "📋  PROJECT INFO"])

# ════════════════════════════════════════════
# TAB 1 — DETECTION
# ════════════════════════════════════════════
with tab_detect:

    st.markdown('<p class="section-label">📁 Upload Satellite Imagery</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a satellite image (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div style="
            text-align:center;padding:48px 24px;
            background:rgba(13,27,46,0.6);
            border:1px dashed rgba(28,58,94,0.8);
            border-radius:4px;margin-top:12px">
          <p style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#1c3a5e">
            NO IMAGERY LOADED
          </p>
          <p style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#5a7a9a;letter-spacing:0.15em">
            UPLOAD A SATELLITE IMAGE TO BEGIN DETECTION
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        image_pil = Image.open(uploaded).convert("RGB")
        w, h = image_pil.size

        # ── RUN INFERENCE ──
        with st.spinner("⚙️ Running inference..."):
            t0 = time.perf_counter()
            annotated_pil, detections = run_inference(model, image_pil, conf_threshold)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── COUNTS ──
        counts = {"airplane": 0, "ship": 0, "vehicle": 0}
        for d in detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        avg_conf = (sum(d["confidence"] for d in detections) / len(detections)) if detections else 0.0

        # ── METRICS ──
        st.markdown('<p class="section-label">📊 Detection Summary</p>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("TOTAL", len(detections))
        m2.metric("✈️ AIRPLANES", counts["airplane"])
        m3.metric("🚢 SHIPS", counts["ship"])
        m4.metric("🚗 VEHICLES", counts["vehicle"])
        m5.metric("AVG CONF", f"{avg_conf:.2f}")
        m6.metric("INFER TIME", f"{elapsed_ms:.0f} ms")

        st.markdown("---")

        # ── SIDE-BY-SIDE IMAGES ──
        st.markdown('<p class="section-label">🖼️ Visual Comparison</p>', unsafe_allow_html=True)
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**ORIGINAL IMAGE**")
            st.image(image_pil, use_container_width=True)
            st.caption(f"Resolution: {w} × {h} px")
        with col_det:
            st.markdown("**DETECTED OBJECTS**")
            st.image(annotated_pil, use_container_width=True)
            st.caption(f"{len(detections)} object(s) detected  |  threshold: {conf_threshold:.2f}")

        st.markdown("---")

        # ── ANALYTICS ──
        if detections:
            st.markdown('<p class="section-label">📈 Visual Analytics</p>', unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)

            with ch1:
                st.markdown("**OBJECT COUNT BY CLASS**")
                bar = make_bar_chart(counts)
                st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

            with ch2:
                st.markdown("**CONFIDENCE DISTRIBUTION**")
                hist = make_conf_hist(detections)
                if hist:
                    st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")

            # ── DETECTION TABLE ──
            st.markdown('<p class="section-label">🗂️ Detection Log</p>', unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame([
                {
                    "ID": i + 1,
                    "Class": d["class"].capitalize(),
                    "Confidence": f"{d['confidence']:.4f}",
                    "Status": "CONFIRMED" if d["confidence"] >= 0.5 else "PROBABLE",
                }
                for i, d in enumerate(detections)
            ])
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "ID":         st.column_config.NumberColumn(width="small"),
                    "Class":      st.column_config.TextColumn(width="medium"),
                    "Confidence": st.column_config.TextColumn(width="medium"),
                    "Status":     st.column_config.TextColumn(width="medium"),
                },
            )

        st.markdown("---")

        # ── DOWNLOADS ──
        st.markdown('<p class="section-label">⬇️ Export Results</p>', unsafe_allow_html=True)
        dl1, dl2, _ = st.columns([1, 1, 2])
        with dl1:
            st.download_button(
                label="⬇ DOWNLOAD DETECTED IMAGE",
                data=pil_to_bytes(annotated_pil),
                file_name="sentinelvision_detection.png",
                mime="image/png",
            )
        with dl2:
            st.download_button(
                label="⬇ DOWNLOAD SUMMARY CSV",
                data=build_csv(detections),
                file_name="sentinelvision_summary.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════
# TAB 2 — PROJECT INFO
# ════════════════════════════════════════════
with tab_info:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("## PROJECT OVERVIEW")
        st.markdown("""
        **SentinelVision AI** is an AI-powered defence surveillance system designed to detect
        defence-relevant objects from satellite imagery using a state-of-the-art oriented
        bounding box object detection model.

        The system targets three high-value object categories commonly analysed in
        intelligence, surveillance, and reconnaissance (ISR) operations.
        """)

        st.markdown("---")
        st.markdown("### ARCHITECTURE DETAILS")
        details = {
            "Model":           "YOLO11s-OBB (Oriented Bounding Box)",
            "Framework":       "Ultralytics YOLO + PyTorch",
            "Dataset":         "DIOR-R (Filtered 3-class)",
            "Epochs":          "50",
            "Image Size":      "640 × 640 px",
            "Batch Size":      "4",
            "GPU":             "NVIDIA Tesla T4",
            "Annotation":      "YOLO OBB (8-point polygon)",
        }
        for k, v in details.items():
            st.markdown(f"""
            <div class="perf-row">
              <span>{k}</span><span class="perf-val">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("### PERFORMANCE METRICS")
        st.markdown("""
        <div class="perf-row"><span>Overall Precision</span><span class="perf-val">92.9%</span></div>
        <div class="perf-row"><span>Overall Recall</span><span class="perf-val">87.7%</span></div>
        <div class="perf-row"><span>mAP@50</span><span class="perf-val">92.9%</span></div>
        <div class="perf-row"><span>mAP@50-95</span><span class="perf-val">77.1%</span></div>
        """, unsafe_allow_html=True)

        st.markdown("### INFERENCE SPEED")
        st.markdown("""
        <div class="perf-row"><span>Preprocess</span><span class="perf-val">0.4 ms</span></div>
        <div class="perf-row"><span>Inference</span><span class="perf-val">7.8 ms</span></div>
        <div class="perf-row"><span>Postprocess</span><span class="perf-val">4.6 ms</span></div>
        <div class="perf-row"><span>Total</span><span class="perf-val">~12.8 ms</span></div>
        """, unsafe_allow_html=True)

        st.markdown("### CLASS-WISE RESULTS")
        for cls, (prec, rec, map50) in {
            "Airplane": ("97.3%", "97.3%", "98.4%"),
            "Ship":     ("95.0%", "96.3%", "98.2%"),
            "Vehicle":  ("86.4%", "69.6%", "82.1%"),
        }.items():
            st.markdown(f"""
            <div style="margin:8px 0 4px;font-family:'Orbitron',monospace;font-size:0.65rem;color:#ffb300">{cls.upper()}</div>
            <div class="perf-row" style="font-size:0.68rem"><span>Precision</span><span class="perf-val">{prec}</span></div>
            <div class="perf-row" style="font-size:0.68rem"><span>Recall</span><span class="perf-val">{rec}</span></div>
            <div class="perf-row" style="font-size:0.68rem"><span>mAP50</span><span class="perf-val">{map50}</span></div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### DATASET INFORMATION")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Original Classes", "20")
        st.metric("Total Images", "23,463")
    with d2:
        st.metric("Filtered Classes", "3")
        st.metric("Dataset Split", "Train / Val / Test")
    with d3:
        st.metric("OBB Format", "8-point polygon")
        st.metric("Label Format", "YOLO OBB")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#5a7a9a;text-align:center;letter-spacing:0.15em">
    SENTINELVISION AI — DEFENCE SATELLITE OBJECT DETECTION SYSTEM &nbsp;│&nbsp;
    YOLO11s-OBB &nbsp;│&nbsp; DIOR-R DATASET &nbsp;│&nbsp; BUILT WITH ULTRALYTICS + STREAMLIT
    </div>
    """, unsafe_allow_html=True)
