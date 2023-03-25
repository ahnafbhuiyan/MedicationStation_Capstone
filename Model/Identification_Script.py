import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import cv2

# Load the saved model
model = tf.keras.models.load_model('pill_identification_model.h5',compile=False)

# Define the input image size
batch_size = 32
img_height = 180
img_width = 180

# Load and preprocess the image
class_names = ['Amlodipine', 'Amoxicillin', 'Atorvastatin', 'Ibuprofen', 'Levothyroxine', 'Lisinopril', 'Losartan', 'Metformin', 'Metoprolol', 'Naproxen', 'Omeprazole', 'Tylenol']

while True:
    press = input("Press space to take a picture")
    if press == "":
        print("Taking Picture")
        cap = cv2.VideoCapture(1)

        # Read a frame from the webcam
        ret, frame = cap.read()
        print(type(frame))
        # Resize the frame to match the input image size
        resized_frame = cv2.resize(frame, (img_width, img_height))
        
        # Convert the frame to a PIL image
        # pill_path = PIL.Image.fromarray(resized_frame)
        # img = np.array(pill_path)
        pill_path = 'OpenCVImg\pic.jpg'
        cv2.imwrite(pill_path,resized_frame)
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

        # # Exit if the 'q' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()