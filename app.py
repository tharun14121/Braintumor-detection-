import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ======================================
# App Configuration
# ======================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title(" Brain Tumor Detection from MRI")
st.write("Upload an MRI image to detect the presence of a brain tumor.")

# ======================================
# Google Drive Model Download
# ======================================
MODEL_URL = "https://drive.google.com/uc?id=1GbedAJ7ANrPkpDWaJlJqdV_UccQfp1p0"
MODEL_PATH = "brain_tumor_model_deploy.keras"

# Remove corrupted / partial file if it exists
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 10_000_000:
    os.remove(MODEL_PATH)

# Download model
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)







# ======================================
# Load Model (Inference only)
# ======================================
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH, compile=False)

model = load_trained_model()

# Class labels (MUST match training order)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ======================================
# Image Preprocessing
# ======================================
IMAGE_SIZE = 224

def preprocess_image(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ======================================
# File Upload
# ======================================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("🔍 Detect Tumor"):
        with st.spinner("Analyzing MRI image..."):
            processed_image = preprocess_image(image)
            preds = model.predict(processed_image)

            pred_idx = np.argmax(preds)
            confidence = np.max(preds)
            label = class_names[pred_idx]

        # ======================================
        # Display Result
        # ======================================
        if label == "notumor":
            st.success(
                f"No Tumor Detected\n\nConfidence: {confidence * 100:.2f}%"
            )
        else:
            st.error(
                f" Tumor Detected: **{label.upper()}**\n\nConfidence: {confidence * 100:.2f}%"
            )
