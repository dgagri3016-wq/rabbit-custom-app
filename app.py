import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Rabbit Weight Predictor",
    page_icon="🐇",
    layout="centered"
)

st.title("🐇 Rabbit Weight Prediction System")
st.markdown(
    "Upload a rabbit image for automatic breed classification, then provide the "
    "segmented mask area (in pixels) to estimate the rabbit's weight."
)

# ==========================================
# MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_weight_pipeline():
    """Loads the Random Forest + OneHotEncoder pipeline."""
    return joblib.load('rabbit_weight_pipeline.joblib')

@st.cache_resource
def load_breed_model():
    """Loads your trained Keras model for breed prediction."""
    try:
        # Update this filename if your Keras model has a different name
        return tf.keras.models.load_model('rabbit_breed_model.keras')
    except Exception as e:
        return None

weight_pipeline = load_weight_pipeline()
breed_model = load_breed_model()

# List of breed labels matching your Keras model's output indices
BREED_CLASSES = ["Holland lop", "Mini rex", "New zealand white"]


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def preprocess_and_predict_breed(image, model):
    """
    Preprocesses the uploaded image and passes it to the Keras breed model.
    Adjust image resizing (e.g., 224x224) to match your Keras training setup.
    """
    if model is None:
        return None
    
    # Resize image to match Keras input specs (e.g., 224x224 or 128x128)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize if model was trained with 0-1 scaling
    
    # Ensure 3 channels (RGB)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
        
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Model inference
    predictions = model.predict(img_batch)
    predicted_class_idx = np.argmax(predictions[0])
    
    return BREED_CLASSES[predicted_class_idx]


# ==========================================
# USER INPUT INTERFACE
# ==========================================
st.subheader("1. Input Data")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Rabbit Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    mask_area = st.number_input(
        "Mask Area (Pixels from YOLO)",
        min_value=1000.0,
        max_value=150000.0,
        value=35000.0,
        step=500.0,
        help="The total pixel area extracted from your YOLOv8 segmentation mask."
    )

# Fallback breed selection if Keras model file is missing or image is not uploaded
manual_breed = None
if breed_model is None or uploaded_file is None:
    manual_breed = st.selectbox(
        "Select Breed (Manual Fallback)",
        options=BREED_CLASSES,
        help="Used if no image is uploaded or if the breed model isn't loaded."
    )


# ==========================================
# PREDICTION LOGIC
# ==========================================
st.markdown("---")

if st.button("Predict Weight", type="primary", use_container_width=True):
    detected_breed = None
    
    with st.spinner("Processing prediction..."):
        # Step 1: Determine Breed (Automated via Keras or Manual Fallback)
        if uploaded_file is not None and breed_model is not None:
            detected_breed = preprocess_and_predict_breed(image, breed_model)
        else:
            detected_breed = manual_breed
            
        # Step 2: Format data for the Weight Pipeline
        input_data = pd.DataFrame([{
            'mask_area': mask_area,
            'breed': detected_breed
        }])
        
        # Step 3: Estimate Weight using the Scikit-Learn Pipeline
        predicted_weight = weight_pipeline.predict(input_data)[0]

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    st.subheader("2. Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Detected / Selected Breed",
            value=detected_breed
        )
        
    with res_col2:
        st.metric(
            label="Predicted Weight",
            value=f"{predicted_weight:.2f} kg"
        )
