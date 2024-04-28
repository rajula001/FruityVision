# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.utils import img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import os

# Set paths to data
train_dir = "./data/train/"
test_dir = "./data/test/"

# Set parameters
img_width, img_height = 100, 100
batch_size = 32
epochs = 20
num_classes = len(os.listdir(train_dir))

# Create data generators
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

# Create model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
# ])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs,
          validation_data=test_generator,
          validation_steps=test_generator.samples // batch_size)

# Test model
loss, accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy}")

# Save model
model.save("classifier.h5")
print("Model saved successfully.")

# Load model
loaded_model = load_model("classifier.h5")
print("Model loaded successfully.")
loss, accuracy = loaded_model.evaluate(test_generator)
print(f"Test accuracy: {accuracy}")
predicted = loaded_model.predict('captured_image.png')
print(predicted)