import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Defence-Relevant Object Detection",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Defence-Relevant Object Detection in Satellite Imagery")

st.markdown("""
Upload a satellite image to detect:
- Airplanes
- Ships
- Vehicles
""")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        results = model.predict(
            source=tmp.name,
            conf=0.25,
            save=False
        )

    result_img = results[0].plot()

    with col2:
        st.subheader("Detection Result")
        st.image(result_img, use_container_width=True)

    st.subheader("Detected Objects")

    boxes = results[0].boxes

    if len(boxes):
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            st.write(
                f"{model.names[cls]} : {conf:.2%}"
            )
    else:
        st.warning("No objects detected.")

    os.unlink(tmp.name)
