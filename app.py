```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Defence-Relevant Object Detection",
    page_icon="🛰️",
    layout="wide"
)

# Title
st.title("🛰️ Defence-Relevant Object Detection in Satellite Imagery")
st.markdown("""
Detect **Airplanes**, **Ships**, and **Vehicles** from satellite imagery using a trained **YOLO11s-OBB** model.
""")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        # Run prediction
        results = model.predict(
            source=tmp.name,
            conf=0.25,
            save=False
        )

    # Plot prediction image
    predicted_image = results[0].plot()

    with col2:
        st.subheader("Detection Results")
        st.image(predicted_image, use_container_width=True)

    # Display detected objects
    st.subheader("Detected Objects")

    boxes = results[0].boxes

    if len(boxes) > 0:

        detected_classes = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            detected_classes.append(
                f"• {class_name} ({confidence:.2f})"
            )

        for item in detected_classes:
            st.write(item)

    else:
        st.warning("No objects detected.")

    # Delete temp file
    os.unlink(tmp.name)

st.markdown("---")
st.markdown(
    "Developed using YOLO11s-OBB, DIOR-R Dataset, Streamlit, and Deep Learning."
)
```
