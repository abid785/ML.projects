import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# App title and layout
st.set_page_config(page_title="🎨 Photo to Sketch & Cartoon", layout="centered")
st.title("🖼️ AI Photo to Sketch & Cartoon App")
st.markdown("Upload a photo and convert it into a sketch or cartoon using OpenCV.")

# Function to convert image to sketch
def convert_to_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch

# Function to convert image to cartoon
def convert_to_cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# Upload image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

# Convert button is inside this block so image updates every time
if uploaded_file:
    # Read and display image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    st.image(image, caption='Original Image', use_container_width=True)

    # Select effect
    option = st.selectbox("Choose a style", ["Sketch", "Cartoon"])

    if st.button("Convert"):
        if option == "Sketch":
            result = convert_to_sketch(img_bgr)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result = convert_to_cartoon(img_bgr)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Display result
        st.image(result_rgb, caption=f'{option} Image', use_container_width=True)

        # Download button
        img_pil = Image.fromarray(result_rgb)
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="📥 Download Image", data=byte_im,
                           file_name=f"{option.lower()}_image.png", mime="image/png")
