import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os
import uuid

# CSS styling
st.markdown("""
    <style>
        .subheader {
            font-size: 18px;
            color: #0A9396 ; /* Change this color as needed */
        }
        .title {
            font-size: 26px;
            color: #0A9396  ; /* Change this color as needed */
            font-weight: bold;
        }
        .sidebar-title {
            font-size: 24px;
            color: #00FA9A ; /* Change this color as needed */
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<div class="title">Face Recognition Attendance System</div>', unsafe_allow_html=True)

# Subheader with styling
st.markdown('<div class="subheader">Either Open Camera & Detect Faces or Upload Images</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    if st.button("Home"):
        st.session_state.page = "home"

# Define the face detection model path
cascade_path = "C:\\Users\\mohammed naser\\Downloads\\deep learning + gen AI\\haarcascade_frontalface_default.xml"
model = cv2.CascadeClassifier('cascade_path')

# Directory to store detected faces
faces_dir = 'detected_faces/'
os.makedirs(faces_dir, exist_ok=True)

# Function to store detected faces
def save_face(image, face_coordinates):
    x, y, w, h = face_coordinates
    face = image[y:y+h, x:x+w]
    face_path = os.path.join(faces_dir, f"{uuid.uuid4()}.jpg")
    cv2.imwrite(face_path, face)
    st.success(f"Face saved to {face_path}")

# Detect faces in an uploaded image
def detect_faces_in_image(uploaded_image):
    img_array = np.array(Image.open(uploaded_image))
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
        save_face(img_array, (x, y, w, h))

    st.image(img_array, channels="BGR", use_column_width=True)

# Detect faces in real-time from the webcam, stop after one face is saved
def detect_faces_in_camera():
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()
    face_saved = False

    while not face_saved:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            save_face(frame, (x, y, w, h))
            face_saved = True
            break  # Stop processing after saving the first face

        st_frame.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
    st.success("Face captured and saved. Camera closed.")

# Button to open the camera for live face detection
if st.button("Open Camera"):
    detect_faces_in_camera()

# File uploader to detect faces in an uploaded image
uploaded_image = st.file_uploader("Upload Image to detect", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
