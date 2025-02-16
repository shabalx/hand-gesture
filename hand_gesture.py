import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import random
import string
import os

# Load Haar Cascade for Face Detection
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(FACE_CASCADE_PATH):
    st.error("Face cascade file missing! Download 'haarcascade_frontalface_default.xml'.")
    st.stop()

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Verify if cascade loaded correctly
if face_cascade.empty():
    st.error("Error loading face cascade. Ensure 'haarcascade_frontalface_default.xml' is valid.")
    st.stop()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Function to generate CAPTCHA
def generate_captcha():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# Initialize session state
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None
if "captcha" not in st.session_state:
    st.session_state["captcha"] = generate_captcha()
if "verified" not in st.session_state:
    st.session_state["verified"] = False
if "hand_verified" not in st.session_state:
    st.session_state["hand_verified"] = False

# If already verified, show success message
if st.session_state["verified"]:
    st.title("🎉 Access Granted! 🎉")
    st.success("Face, Hand, and CAPTCHA verified! You are now authenticated.")
    st.stop()

st.title("Face & Hand Gesture CAPTCHA Verification")
st.write("Please allow webcam access to proceed.")

# Capture Image from Webcam
if st.button("Capture Image", key="capture_btn"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.resize(frame, (640, 480))
        st.session_state["captured_image"] = frame
    else:
        st.error("Failed to capture image.")

# Process Captured Image
if st.session_state["captured_image"] is not None:
    captured_image = st.session_state["captured_image"]
    st.image(captured_image, channels="BGR", caption="Captured Image")

    # Convert to grayscale for face detection
    captured_gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    # Detect Face
    detected_faces = face_cascade.detectMultiScale(captured_gray, scaleFactor=1.1, minNeighbors=4)

    # Convert image to RGB for MediaPipe
    captured_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Hands
    results = hands.process(captured_rgb)

    # Check for detected face
    if len(detected_faces) > 0:
        st.success("✅ Face detected!")
    else:
        st.warning("❌ No face detected. Please try again.")

    # Check for detected hand
    if results.multi_hand_landmarks:
        st.success("✅ Hand detected!")
        st.session_state["hand_verified"] = True
        
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(captured_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display updated image with hand landmarks
        st.image(captured_image, channels="BGR", caption="Processed Image with Hand Detection")
    else:
        st.warning("❌ No hand detected. Please try again.")

# CAPTCHA Verification
st.subheader("Enter CAPTCHA to Verify")
st.write(f"Your CAPTCHA: **{st.session_state['captcha']}**")
user_captcha = st.text_input("Enter the CAPTCHA above")

if st.button("Verify CAPTCHA", key="verify_btn"):
    if user_captcha == st.session_state["captcha"] and st.session_state["hand_verified"]:
        st.success("✅ Verification successful!")
        st.session_state["verified"] = True
    else:
        st.error("❌ CAPTCHA incorrect or hand not detected. Try again.")
