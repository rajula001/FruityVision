import os
import cv2 as cv
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import csv

# Load the model
model = load_model("classifier.h5")

# Get labels
labels = sorted(os.listdir('./data/train/'))

# Preprocess image
def preprocess(img_path, target_size=(100, 100)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Identify fruits
def identify(img):
    img_array = preprocess(img)
    predicted = model.predict(img_array)
    predicted_index = np.argmax(predicted)
    predicted_label = labels[predicted_index]
    return predicted_label

# Define the file path
file_path = "nutritional_information.csv"

# Function to find matching fruits nutrition
def get_matching_fruits_nutrition(fruit_name):
    matching_fruits = {}
    for name, data in nutrition_data.items():
        if fruit_name.lower() in name:
            matching_fruits[name] = data
    return matching_fruits

# Initialize an empty dictionary to store the nutrition data
nutrition_data = {}

# Open the CSV file and read its contents using DictReader
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        food_name = row.pop('name').lower()  # Convert to lowercase for case insensitivity
        if food_name not in nutrition_data:
            nutrition_data[food_name] = row

# Open webcam
cap = cv.VideoCapture(0)

output_dict = {}

detected_fruit = None

while True:
    ret, frame = cap.read()
    
    cv.imshow("Webcam", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite('captured_img.png', frame)
        detected_fruit = identify('captured_img.png')
        print(f"Fruit identified as {detected_fruit}. Getting nutritional info...")
        matching_fruits_nutrition = get_matching_fruits_nutrition(detected_fruit)
        if matching_fruits_nutrition:
            output_dict[detected_fruit] = matching_fruits_nutrition
        else:
            output_dict[detected_fruit] = "No nutritional information found! :("
        print(f"Nutritional information of {detected_fruit}:")
        print(output_dict)

# Release webcam and close all windows
cap.release()
cv.destroyAllWindows()
