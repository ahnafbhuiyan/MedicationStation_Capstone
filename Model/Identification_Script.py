import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import cv2
import serial

ser = serial.Serial('COM7', 9600,timeout=1)

# Load the saved model
model = tf.keras.models.load_model('pill_identification_model_SMALL.h5',compile=False)

# Define the input image size
batch_size = 32
img_height = 180
img_width = 180

# Load and preprocess the image
class_names = ['Naproxen', 'Tylenol']
#class_names = ['Amlodipine', 'Amoxicillin', 'Atorvastatin', 'Ibuprofen', 'Levothyroxine', 'Lisinopril', 'Losartan', 'Metformin', 'Metoprolol', 'Naproxen', 'Omeprazole', 'Tylenol']

cap = cv2.VideoCapture(1)


while True:
    input("Press Enter to start")
    ser.write(b's')
    if ser.in_waiting > 0:
        print("Message Recieved")
        message = ser.readline().decode()# Read the serial signal
        if message == "Done\r\n":
            print("Stepper motor has completed its rotation.") 
            # Read a frame from the webcam
            ret, frame = cap.read()
            # Resize the frame to match the input image size
            #resized_frame = cv2.resize(frame, (img_width, img_height))
            cropped = frame[170:300, 250:380]
            pill_path = 'OpenCVImg\pic.jpg'
            cv2.imwrite(pill_path,cropped)
            img = tf.keras.utils.load_img(
                pill_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

