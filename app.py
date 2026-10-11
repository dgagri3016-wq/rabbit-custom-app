import io
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================

BREED_MODEL_URL = "https://github.com/dgagri3016-wq/rabbit-custom-app/releases/download/breed-model/rabbit_breed_final_model.keras"

BREED_MODEL_PATH = "rabbit_breed_final_model.keras"
WEIGHT_PIPELINE_PATH = "rabbit_weight_pipeline.joblib"
LABELS_PATH = "tlabels.txt"

BREED_IMG_SIZE = (299, 299)
DEFAULT_MASK_AREA = 35000.0  # Constant passed behind the scenes for weight estimation

# Page configuration
st.set_page_config(page_title="Rabbit AI Predictor", page_icon="🐇", layout="centered")

st.markdown("<h2 style='text-align: center;'>🐇 Rabbit AI Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Take or upload a photo to instantly analyze your rabbit</p>", unsafe_allow_html=True)

# ==========================================
# 2. MODEL & FILE LOADERS
# ==========================================

@st.cache_resource
def load_breed_model():
    """Downloads Keras breed model from GitHub Release if missing."""
    if os.path.exists(BREED_MODEL_PATH) and os.path.getsize(BREED_MODEL_PATH) < 1000000:
        os.remove(BREED_MODEL_PATH)
            
    if not os.path.exists(BREED_MODEL_PATH):
        st.info("Downloading Keras Breed Model from GitHub Release... Please wait.")
        urllib.request.urlretrieve(BREED_MODEL_URL, BREED_MODEL_PATH)
        
    return tf.keras.models.load_model(BREED_MODEL_PATH)

@st.cache_resource
def load_weight_pipeline():
    """Loads the Scikit-Learn weight pipeline directly from repo main folder."""
    if not os.path.exists(WEIGHT_PIPELINE_PATH):
        st.error(f"Missing '{WEIGHT_PIPELINE_PATH}'. Ensure it is committed directly to your repository main branch.")
        st.stop()
    return joblib.load(WEIGHT_PIPELINE_PATH)

@st.cache_data
def load_labels():
    """Loads breed labels from tlabels.txt."""
    if not os.path.exists(LABELS_PATH):
        return None
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def preprocess_image_for_breed(image):
    image = image.convert("RGB")
    image = image.resize(BREED_IMG_SIZE)
    img_array = np.array(image)
    return np.expand_dims(img_array, axis=0)

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

class_labels = load_labels()
if not class_labels:
    st.error(f"Missing {LABELS_PATH}. Ensure tlabels.txt is in your repo main folder.")
    st.stop()

if "saved_image" not in st.session_state:
    st.session_state.saved_image = None

# Default selection set to "Camera Capture"
option = st.selectbox(
    "📸 Choose image source:", 
    ("Camera Capture", "Upload Image")
)

if option == "Camera Capture":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        st.session_state.saved_image = Image.open(camera_file)
    else:
        st.session_state.saved_image = None

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Select a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state.saved_image = Image.open(uploaded_file)
        st.image(st.session_state.saved_image, caption="Uploaded Image", use_container_width=True)
    else:
        st.session_state.saved_image = None

# ==========================================
# 5. AUTOMATIC PREDICTION & RESULTS DISPLAY
# ==========================================

if st.session_state.saved_image is not None:
    with st.spinner("Analyzing image and calculating weight..."):
        
        # --- BREED PREDICTION ---
        breed_model = load_breed_model()
        breed_processed = preprocess_image_for_breed(st.session_state.saved_image)
        breed_predictions = breed_model.predict(breed_processed)
        
        predicted_index = np.argmax(breed_predictions[0])
        confidence = breed_predictions[0][predicted_index]
        predicted_label = class_labels[predicted_index]
        
        st.divider() 
        
        col1, col2 = st.columns(2)
        col1.metric("Primary Breed", predicted_label)
        col2.metric("Confidence", f"{confidence * 100:.1f}%")
        
        # --- WEIGHT PREDICTION (Joblib Pipeline) ---
        try:
            weight_pipeline = load_weight_pipeline()
            
            input_df = pd.DataFrame([{
                'mask_area': DEFAULT_MASK_AREA,
                'breed': predicted_label
            }])
            
            predicted_weight = weight_pipeline.predict(input_df)[0]
            
            st.metric("⚖️ Estimated Weight", f"{predicted_weight:.2f} kg")
        except Exception as e:
            st.error(f"Could not calculate weight: {e}")

        # --- PROBABILITY BREAKDOWN ---
        with st.expander("📊 View Top 3 Breed Probabilities"):
            top_indices = np.argsort(breed_predictions[0])[-3:][::-1]
            for i in top_indices:
                st.write(f"**{class_labels[i]}**")
                st.progress(float(breed_predictions[0][i]))
