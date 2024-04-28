# FruityVision
FruityVision is a project that utilizes computer vision and deep learning to identify fruits from images or live camera feeds and provides instant nutritional information about them. This project consists of three main components: Backend, CNN (Convolutional Neural Network), and Fruit-detector.

## Backend
The Backend component includes the code responsible for loading the pre-trained classification model, preprocessing images, identifying fruits, retrieving matching nutritional information from a CSV file, and interacting with the webcam to capture images for classification.

### Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Keras
- Pandas

## CNN
The CNN component contains the code for training a convolutional neural network (CNN) model to classify fruits. It uses the Keras library for building and training the model.

### Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- OpenCV (`cv2`)
- Keras

## Fruit-detector
The Fruit-detector component is a Streamlit application that provides a user-friendly interface for interacting with the FruityVision system. It allows users to toggle the camera, capture images of fruits, upload images for classification, and view nutritional information about the identified fruits.

### Requirements
- Python 3.x
- OpenCV (`cv2`)
- Streamlit
- Pandas
- NumPy
- Keras

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies listed in each component's section.
3. Run the Fruit-detector component using Streamlit.
4. Follow the instructions provided in the application to interact with the FruityVision system.

## Acknowledgments
- The classification model was trained on a dataset containing images of various fruits.
- Nutritional information was sourced from the provided CSV file.

For detailed usage instructions and additional information, refer to the documentation included with each component.


