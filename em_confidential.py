import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import os
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image
import base64
import io

# Set up the Streamlit app
st.set_page_config(page_title="Paldron: EM Live Redaction", layout="centered")
st.image("mylog.png", width=150)
st.title("Paldron: EM Live Redaction")

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize EasyOCR reader with GPU
reader = easyocr.Reader(['en'], gpu=True)

# Function to redact text in an image
def redact_text(image, ocr_results, allowed_words):
    for (bbox, text, _) in ocr_results:
        if text.lower() not in allowed_words:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=-1)
    return image

# Streamlit app main function
def main():
    allowed_words_input = st.text_input("Enter words to NOT redact (comma-separated)", "")
    allowed_words = allowed_words_input.lower().split(',')
    
    # Controls for webcam feed
    if st.button("Start Webcam"):
        # Start webcam capture
        cap = cv2.VideoCapture(0)
        
        st.session_state.webcam_started = True
        frame_window = st.image([])

        while st.session_state.webcam_started:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ocr_results = reader.readtext(frame_rgb)
            redacted_frame = redact_text(frame_rgb, ocr_results, allowed_words)
            
            # Update Streamlit display
            frame_window.image(redacted_frame)

        cap.release()

    if st.button("Stop Webcam"):
        st.session_state.webcam_started = False

    # Controls for image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        ocr_results = reader.readtext(image_cv)
        redacted_image = redact_text(image_cv, ocr_results, allowed_words)
        
        # Display the original and redacted images
        st.image(image, caption="Uploaded Image")
        st.image(redacted_image, caption="Redacted Image")
        
        # Allow download of redacted image
        result_image = Image.fromarray(cv2.cvtColor(redacted_image, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(label="Download Redacted Image",
                           data=byte_im,
                           file_name="redacted_image.png",
                           mime="image/png")

if __name__ == "__main__":
    if 'webcam_started' not in st.session_state:
        st.session_state.webcam_started = False
    main()
