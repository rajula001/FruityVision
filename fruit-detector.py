import os
import cv2
import streamlit as st
import pandas as pd
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import time
import csv

# Nutritional information file
info_file = "./nutritional_information.csv"
df = pd.read_csv(info_file)
info_headers = list(df.columns.values)[1:]

st.set_page_config(layout="wide")

# Set app header
st.image('fruit.png', width=100)
st.title("FruityVision")
st.write("Get instant nutritional info!")
st.write("How to use: Toggle camera and take a picture of your fruit to find out its nutritional information.")
st.write("IMPORTANT: Hold the fruit close to the camera and within the green box for better results! :)")

col1, col2 = st.columns((1, 1), gap="large")

# Initialize an empty dictionary to store the nutrition data
if 'nutrition_data' not in st.session_state:
    st.session_state.nutrition_data = {}
    # Open the CSV file and read its contents using DictReader
    with open(info_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            food_name = row.pop('name').lower()  # Convert to lowercase for case insensitivity
            if food_name not in st.session_state.nutrition_data:
                st.session_state.nutrition_data[food_name] = row

# Load the model
if 'model' not in st.session_state:
    st.session_state.model = load_model("classifier.h5", compile=False)

# Get labels
if 'labels' not in st.session_state:
    st.session_state.labels = sorted(os.listdir('./data/train/'))

if 'predicted_fruit' not in st.session_state:
    st.session_state.predicted_fruit = None

if 'frame_window' not in st.session_state:
    st.session_state.frame_window = None

if 'roi' not in st.session_state:
    st.session_state.roi = None

def save_frame():
    st.session_state.captured_frame = st.session_state.frame

# Preprocess image
def preprocess(img_path, target_size=(100, 100)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Identify fruits
def identify(img):
    img_array = preprocess(img)
    predicted = st.session_state.model.predict(img_array)
    predicted_index = np.argmax(predicted)  # Pick index of most likely label
    predicted_label = st.session_state.labels[predicted_index]  # Get name of label
    return predicted_label

with col1:
    
    col1_1, col1_2 = st.columns((1,1))
    
    with col1_1:
        # Initialize a session state variable to keep track of the button state
        if 'toggle_button_state' not in st.session_state:
            st.session_state.toggle_button_state = False

        # Toggle button
        if st.button("Toggle camera", use_container_width=True):
            st.session_state.toggle_button_state = not st.session_state.toggle_button_state

        # Toggle camera on or off
        if st.session_state.toggle_button_state:
            run = True
        else:
            run = False
                
    with col1_2:
        
        # If camera is running
        if run:
            if st.button(label="Take picture", disabled=False, use_container_width=True, on_click=save_frame):
                # Save image of ROI for classification
                st.session_state.roi = cv2.cvtColor(st.session_state.roi, cv2.COLOR_BGR2RGB)
                cv2.imwrite('captured_roi.png', st.session_state.roi)
                
                # Identify fruit in frame
                st.session_state.predicted_fruit = identify('captured_roi.png')
                run = False
                # Toggle camera off
                st.session_state.toggle_button_state = not st.session_state.toggle_button_state
                
                # Display captured image in frame
                st.session_state.frame_window.image(st.session_state.captured_frame, caption="Captured image", use_column_width=True)
                
                # Display nutritional information  
                with col2:
                        
                    st.header("Nutritional information")
                    
                    st.subheader(f"Identified fruit: {st.session_state.predicted_fruit}")
                    
                    col2_1, col2_2, col2_3 = st.columns((1, 1, 1))
                    
                    # If fruit is detected
                    if st.session_state.predicted_fruit in list(st.session_state.nutrition_data.keys()):
                        print(st.session_state.nutrition_data[st.session_state.predicted_fruit])
                        
                        with col2_1:
                            for i in range(7):
                                st.write(f"{info_headers[i]}: {st.session_state.nutrition_data[st.session_state.predicted_fruit][info_headers[i]]}")
                        
                        with col2_2:
                            for i in range(7, 14):
                                st.write(f"{info_headers[i]}: {st.session_state.nutrition_data[st.session_state.predicted_fruit][info_headers[i]]}")
                        
                        with col2_3:
                            for i in range(14, 21):
                                st.write(f"{info_headers[i]}: {st.session_state.nutrition_data[st.session_state.predicted_fruit][info_headers[i]]}")
                                
                    else:
                        st.write(f"Sorry, we do not have any nutritional information on {st.session_state.predicted_fruit}! :(")
                            
        else:
            if st.button(label="Take picture", disabled=False, use_container_width=True):
                error = st.error("ERROR: Toggle camera on first.", icon="🚨")
                time.sleep(2)
                error.empty()

        # Upload picture button
        uploaded_file = st.file_uploader("Upload fruit image", type=["jpg", "png"], key="file_uploader")

        if uploaded_file is not None:
            # Save uploaded image
            with open('uploaded_image.png', 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Identify fruit in uploaded image
            st.session_state.predicted_fruit = identify('uploaded_image.png')
            # Display nutritional information  
            with col2:
                st.header("Nutritional information")

                st.subheader(f"Identified fruit: {st.session_state.predicted_fruit}")

                col2_1, col2_2, col2_3 = st.columns((1, 1, 1))

                # If fruit is detected
                if st.session_state.predicted_fruit in list(st.session_state.nutrition_data.keys()):
                    print(st.session_state.nutrition_data[st.session_state.predicted_fruit])

                    with col2_1:
                        for i in range(7):
                            st.write(f"{info_headers[i]}: {st.session_state.nutrition_data[st.session_state.predicted_fruit][info_headers[i]]}")

                    with col2_2:
                        for i in range(7, 14):
                            st.write(f"{info_headers[i]}: {st.session_state.nutrition_data[st.session_state.predicted_fruit][info_headers[i]]}")

                    with col2_3:
                        for i in range(14, 21):
                            st.write(f"{info_headers[i]}: {st.session_state.nutrition_data[st.session_state.predicted_fruit][info_headers[i]]}")

                else:
                    st.write(f"Sorry, we do not have any nutritional information on {st.session_state.predicted_fruit}! :(")

    # Create viewport frame
    if st.session_state.frame_window == None:
        st.session_state.frame_window = st.image([], use_column_width=True)

    # Interface with webcam
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while run:
        _, frame = cam.read()  # Read webcam data

        # Create green bounding box
        bb_left = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) - 200
        bb_top = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) - 200
        bb_right = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2) + 200
        bb_bottom = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) + 200
        cv2.rectangle(frame, (bb_left, bb_top), (bb_right, bb_bottom), (0, 255, 0), 3)

        # Save frame and ROI
        st.session_state.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.roi = st.session_state.frame[bb_top:bb_bottom, bb_left:bb_right]
        st.session_state.frame_window.image(st.session_state.frame, caption="Camera feed", use_column_width=True)
