import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os
import uuid

st.markdown(
    """
    <style>
    /* Global styles */
    body {
        font-family: 'Arial', sans-serif;
    }

    /* Center align the title */
    .title, .header {
        text-align: center;
        color: #33ff33; /* A nice green color */
        font-size: 60px; /* Larger font size */
        margin-top: 20px;
        margin-bottom: 60px; /* Space below the title */
    }
    .hh{
        text-align: center;
        color: #33ff33; /* A nice green color */
        font-size: 30px; /* Larger font size */
        margin-top: 10px;
        margin-bottom: 10px; /* Space below the title */
    }

    /* Style the sidebar */
    .css-1aumxhk {
        background-color: #33ff33;!important; /* Light sidebar background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Style buttons */
    .stButton > button {
        background-color: #800040; /* purple background for buttons */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        color: white;
        background-color: green; /* light pruple on hover */
        border-style: solid;
        border-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<h1 class='title'>Face Recognition Attendance System</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='hh'>Open Camera & Detect Faces</h3>", unsafe_allow_html=True)

# Define the face detection model path
cascade_path = "C:\\Users\\mohammed naser\\Downloads\\deep learning + gen AI\\haarcascade_frontalface_default.xml"

model = cv2.CascadeClassifier('cascade_path')

# Directory to store detected faces
faces_dir = 'detected_faces/'
os.makedirs(faces_dir, exist_ok=True)

# Function to store detected faces with a custom name
def save_face(image, face_coordinates, custom_name):
    x, y, w, h = face_coordinates
    face = image[y:y+h, x:x+w]
    
    # Ensure the custom name has a .jpg extension
    if not custom_name.endswith(".jpg"):
        custom_name += ".jpg"
        
    face_path = os.path.join(faces_dir, custom_name)
    cv2.imwrite(face_path, face)
    st.success(f"Face saved as {custom_name} in {faces_dir}")

# Detect faces in an uploaded image with custom naming for each face


# Detect faces in real-time from the webcam with custom naming
def detect_faces_in_camera(custom_name):
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
            save_face(frame, (x, y, w, h), custom_name)
            face_saved = True
            break  # Stop processing after saving the first face

        st_frame.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
    if face_saved:
        st.success("Face captured and saved. Camera closed.")

# Main interface for face detection

# Button to open the camera for live face detection

custom_name = st.text_input("Enter Your Name") 

if st.button("Open Camera"):           
    detect_faces_in_camera(custom_name)