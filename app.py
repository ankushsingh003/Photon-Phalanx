import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="Photon Phalanx | Defect Classifier",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS for Maximum Visibility - Light Theme
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Outfit:wght@700&display=swap" rel="stylesheet">
<style>
    /* Global Background Fix */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f9f4 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    .main {
        background-color: #f0f9f4 !important;
        color: #000000 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Fix */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #d0eadd !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* Main Container text */
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] h1, 
    [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] span, label {
        color: #000000 !important;
        font-weight: 500;
    }

    /* Header Styling */
    .header-container {
        padding: 2rem 0;
        text-align: center;
    }
    
    .header-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #27ae60 !important;
        font-size: 3.5rem;
        margin-bottom: 0.2rem;
    }

    /* Card Styling - Solid White */
    .prediction-card {
        background-color: #ffffff !important;
        padding: 30px;
        border-radius: 20px;
        border: 2px solid #e0eadd;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        color: #000000 !important;
    }
    
    /* File Uploader visibility */
    .stUploadedFile {
        background-color: #f9fdfb !important;
        border: 2px dashed #2ecc71 !important;
        color: #000000 !important;
    }

    .stButton>button {
        background: #27ae60 !important;
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 10px;
        padding: 15px 40px;
        border: none;
        width: 100%;
    }

    /* Info/Warning Overrides for Light Mode */
    .stAlert {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #e0eadd !important;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    model_path = 'saved_models/best_mobilenet_v2.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3222/3222792.png", width=100)
st.sidebar.title("Photon Phalanx AI")
st.sidebar.markdown("---")
st.sidebar.info("""
**Photon Phalanx AI** uses state-of-the-art Deep Learning to identify solar panel defects.
                
**Current Capabilities:**
- 🛡️ Physical Damage Detection
- ⚡ Electrical Fault Analysis
- ❄️ Snow/Dust Coverage
- 🐦 Bird Drop Scoring
""")

# Main Header with Animation Wrapper
st.markdown("""
<div class="header-container">
    <p class="header-text">Photon Phalanx</p>
    <p style="font-size: 1.3rem; color: #000000; font-weight: 500;">Advanced Computer Vision for Solar Asset Management</p>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("📤 Asset Upload")
    uploaded_file = st.file_uploader("Drop inspection imagery here (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Inspection Frame', use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("🔍 Intelligence Insight")
    if uploaded_file is not None:
        model = load_model()
        if model:
            with st.spinner('Neural Network Inference...'):
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                
                classes = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
                result_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                label = classes[result_idx]
                
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <h2 style="color: #2c3e50; margin-bottom: 0;">{label} Identified</h2>
                    <p style="color: #000000; font-size: 1.1rem;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization
                st.progress(int(confidence))
                
                if label == 'Clean':
                    st.success("✅ Operational Excellence: No defects detected.")
                else:
                    st.warning(f"⚠️ Maintenance Alert: {label} affects yield by ~15%.")
        else:
            # Training Status with Glow
            st.markdown('<div class="training-glow" style="padding: 15px; border-radius: 10px; background: rgba(255, 179, 2, 0.05);">', unsafe_allow_html=True)
            st.warning("🚀 **Model Optimization in Progress**")
            st.info("The neural engine is currently learning from your dataset. This establishes the baseline for defect patterns.")
            
            if os.path.exists("training.log"):
                st.write("---")
                st.caption("Live Training Stream:")
                try:
                    with open("training.log", "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        last_lines = "".join(lines[-10:]) if lines else "Connecting to training process..."
                        st.code(last_lines)
                except:
                    st.code("Initializing logs...")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting input imagery. Upload a frame on the left to begin analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #000000; font-size: 0.9rem; font-weight: bold; margin-top: 50px;">
    Photon Phalanx Pro v1.2 | Enterprise Edge Inference Mode | Built for Sustainable Energy
</div>
""", unsafe_allow_html=True)
