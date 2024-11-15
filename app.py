import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os

st.markdown(
    """
    <style>
    /* Global styles */
    body {
        font-family: 'Arial', sans-serif;
    }

    /* Center align the title */
    .title{
        text-align: center;
        color: #6600cc; /* A nice purple color */
        font-size: 50px; /* Larger font size */
        margin-top: 0px;
        margin-bottom: 0px; /* Space below the title */
    }
    .hh{
        text-align: center;
        color: #6600cc; /* A nice purple color */
        font-size: 30px; /* Larger font size */
        margin-top: 10px;
        margin-bottom: 10px; /* Space below the title */
    }

    /* Style the sidebar */
    .css-1aumxhk {
        background-color: #f0f4f8 !important; /* Light sidebar background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Style buttons */
    .stButton > button {
        background-color: #6600cc; /* purple background for buttons */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        color: white;
        background-color: green; /* light purple on hover */
        border-style: solid;
        border-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Face Detector by Faraz</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='hh'>Detect Faces by Uploading Images</h3>", unsafe_allow_html=True)

# Load the Haar Cascade model
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(cascade_path)

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

# Detect faces in an uploaded image and then allow saving
def detect_faces_in_image(uploaded_image):
    img_array = np.array(Image.open(uploaded_image))
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        st.warning("No faces detected.")
        return None, None

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img_array, channels="BGR", use_column_width=True)
    return img_array, faces

# File uploader to detect faces in an uploaded image
uploaded_image = st.file_uploader("Upload Image to detect", type=["jpg", "png", "jpeg"], key="uploaded_image")
if uploaded_image is not None:
    # Detect face and display it first
    img_array, faces = detect_faces_in_image(uploaded_image)
    if faces is not None:
        if 'custom_name' not in st.session_state:
            st.session_state.custom_name = "face_image"
        
        # Only show input and save button once
        custom_name = st.text_input("Enter a custom name for the uploaded face image", value=st.session_state.custom_name, key="upload_custom_name")
        
        if st.button("Save Detected Face", key="save_uploaded_face"):
            for face_coords in faces:
                save_face(img_array, face_coords, custom_name)

# Button to open the camera for live face detection
st.markdown("<h3 class='hh'>Either Open Camera & Detect Faces</h3>", unsafe_allow_html=True)

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
custom_name = st.text_input("Enter Your Name") 

if st.button("Open Camera"):           
    detect_faces_in_camera(custom_name)

st.write("Please note that camera access permissions are managed by Hugging Face, which is why the 'Open Camera' button may not function as expected.")
st.write("However, the underlying code performs optimally when run in a local environment.")

# Sidebar for displaying the video
st.sidebar.title("Watch the Video of Demo")

# YouTube video link
video_url = "https://www.youtube.com/watch?v=Rw2tE-e-0pY?si=SGMwZZNpbJe-WAa3"  # Replace with your YouTube video URL
st.sidebar.video(video_url)
