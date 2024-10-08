import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the improved trained model
try:
    model = load_model('model/skin_cancer_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def predict_image(image):
    if model is None:
        logging.error("Model is not loaded.")
        return np.array([[0, 0]])
    
    try:
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)
        logging.debug(f"Image shape for prediction: {img.shape}")

        prediction = model.predict(img)
        logging.debug(f"Prediction: {prediction}")
        
        return prediction
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return np.array([[0, 0]])

def app():
    st.title("📸 Skin Cancer Detection")
    st.markdown("""
                **🔍 Ready to check for skin cancer?** 

                Upload a photo to get started. Our advanced model will analyze the image and provide insights into potential skin conditions.
                """)
    with st.expander("See More Details"):
        st.markdown("""
                    ### 📂 Upload an Image
                    Select a saved **skin lesion image** from your device for evaluation. Ensure the image is clear and well-lit for the best results.

                    ### How It Works:
                    - **Upload:** 
                    Easily upload an image for analysis. Make sure the area of interest is visible and clear for accurate results. 📸
                    
                    - **Analyze:** 
                    Our advanced model will process the image and analyze it for potential skin conditions.🔍
                    
                    - **Get Insights:** 
                    Receive immediate feedback on your skin health based on the analysis.💡
                    

                    Ready to begin? Upload an image below and let’s get started! 🚀
                    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((500, 500))
        img_base64 = image_to_base64(image)
        st.markdown(f"""
            <div style='text-align: center;'>
                <img src="data:image/png;base64,{img_base64}" width="500" />
                <div style='margin-top: 10px; font-size: 16px; color: #666;'>🖼️ Uploaded Image</div>
            </div>
            """, unsafe_allow_html=True)
        
        prediction = predict_image(image)
        
        if prediction is not None:
            benign_prob = prediction[0][0]
            malignant_prob = prediction[0][1]
            
            st.write(f"**Benign Probability:** {benign_prob * 100:.2f}%")
            st.write(f"**Malignant Probability:** {malignant_prob * 100:.2f}%")
            
            threshold = 0.5
            
            if malignant_prob > threshold:
                st.markdown(f"""
                    <div style='text-align: center; background-color: #ffcccc; padding: 10px; border-radius: 5px;'>
                        <h3 style='color: #ff0000;'>🚨 Skin Cancer Detected</h3>
                        <p style='color: #0F0F0F;'>Detection: Malignant (Confidence: {malignant_prob * 100:.2f}%)</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='text-align: center; background-color: #ccffcc; padding: 10px; border-radius: 5px;'>
                        <h3 style='color: #006400;'>✅ No Skin Cancer Detected</h3>
                        <p style='color: #0F0F0F;'>Detection: Benign (Confidence: {benign_prob * 100:.2f}%)</p>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
