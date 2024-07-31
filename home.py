import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image, ImageDraw

# Function to load Lottie animation from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Function to round the corners of an image
def round_corners(image, radius):
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    left_up_point = (0, 0)
    right_down_point = (image.size[0], image.size[1])
    draw.rounded_rectangle([left_up_point, right_down_point], radius, fill=255)
    image.putalpha(mask)
    return image

def app():
    # Lottie animation URL
    lottie_url = "https://lottie.host/b1dd8e5a-a564-456a-8b07-ec8ca2844d35/NZRI2uWqny.json"  # Lottie URL , extra->(https://lottie.host/8e7b345a-b604-4323-bb9f-ff397acee353/UwubwemMkg.json)
    lottie_json = load_lottieurl(lottie_url)
    
    # Load the image
    image_path = "images/topbanner.png"
    image = Image.open(image_path)

    # Resize the image while keeping the height constant
    new_width = 1200
    new_height = 400
    resized_image = image.resize((new_width, new_height))
    
    # Round the corners of the image
    rounded_image = round_corners(resized_image, radius=15)

    # Display the resized and rounded image
    st.image(rounded_image)

    # Create two columns
    col1, col2 = st.columns([2, 1])  # Adjust the column width ratio

    # Introduction text in the first column
    with col1:
        st.markdown("""
                    ### 🌟 Welcome to SmartSkin Scan 🌟
                    
                    **SmartSkin Scan** is an innovative application designed to assist in the early detection of skin cancer through advanced AI technology. 
                    Leveraging state-of-the-art machine learning models, our app analyzes images of skin lesions to provide accurate assessments and classifications.
                    
                    **Why Choose SmartSkin Scan?**
                    - **🕵️‍♂️ Early Detection:** Identify potential skin cancer at an early stage with high accuracy.
                    - **😊 User-Friendly:** Simple and intuitive interface for easy image uploads or real-time camera use.
                    - **🤖 Advanced Technology:** Powered by cutting-edge AI and deep learning models to ensure reliable results.
                    - **📊 Comprehensive Insights:** Get detailed analysis and probabilities for benign and malignant conditions.
                    
                    **What You Can Do with SmartSkin Scan:**
                    - **📸 Upload or Capture Images:** Easily upload saved images or use camera for instant analysis.
                    - **⏱️ Real-Time Analysis:** Use our advanced AI model to detect and classify skin conditions from your images.
                    - **🧠 Model Training:** Train the model in real-time to improve its performance and get hands-on experience with machine learning.
                    - **🔍 View Results:** Receive immediate feedback and detailed insights about your skin health.
                    
                    **How It Works:**
                    1. **📤 Upload or Capture:** Choose an image of a skin lesion or use your camera for real-time analysis.
                    2. **🔬 Analysis:** Our AI model processes the image to detect and classify the skin condition.
                    3. **📈 Results:** Receive instant feedback with confidence scores, helping you make informed decisions about your skin health.
                    
                    Your skin health is our priority. Use **SmartSkin Scan** to stay vigilant and proactive in managing your skin's well-being!
                    """)


    # Lottie animation in the second column
    with col2:
        st.markdown('<div style="margin-top: -80px;"></div>', unsafe_allow_html=True)
        if lottie_json:
            st_lottie(lottie_json, speed=1, width=400, height=400, key="intro_animation")  # Adjust width and height

if __name__ == "__main__":
    app()