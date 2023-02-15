# # Import necessary libraries
# import tensorflow as tf
# from tensorflow import keras
# from keras.preprocessing.image import ImageDataGenerator

# # Create a data generator for our dataset
# datagen = ImageDataGenerator(rescale = 1./255)

# # Use the data generator to read in our dataset
# train_data = datagen.flow_from_directory(
#     'D:\Documents\Coding\CAPSTONE\ChatGPT\pillbox_production_images_full_202008',
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical')

# print(len(train_data))
# # Build the model using a pre-trained CNN
# model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# # Add additional layers to the pre-trained model
# x = model.output
# x = keras.layers.Flatten()(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
# x = keras.layers.Dropout(0.5)(x)
# x = keras.layers.Dense(len(train_data.class_indices), activation='softmax')(x)

# # Compile the model
# model = keras.Model(inputs=model.input, outputs=x)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_data, epochs=10)

# # Save the model
# model.save('pill_identification_model.h5')

# # Use the trained model to predict the name of a pill
# from keras.preprocessing import image
# import numpy as np

# img = tf.keras.utils.load_img('D:\Documents\Coding\CAPSTONE\ChatGPT\pillbox_production_images_full_202008\Clonazepam_Tablet\\0.5_mg_Clonazepam_Tablet.jpg', target_size=(150, 150))
# x = tf.keras.utils.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# prediction = model.predict(x)
# print(train_data)
# print(np.argmax(prediction))
# pill_name = train_data.class_indices['Clonazepam_Tablet']

# for name in train_data.class_indices:
#     if train_data.class_indices[name] == np.argmax(prediction):
#         print (name)
# print("This pill is:", pill_name)


# def findPillName(dict, index):
#     for name in dict:
#         if dict[name] == index:
#             return name

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.preprocessing import image

# Create a data generator for our dataset
datagen = ImageDataGenerator(rescale=1./255)

# Use the data generator to read in our dataset
train_data = datagen.flow_from_directory(
    'D:\Documents\Coding\CAPSTONE\ChatGPT\pillbox_production_images_full_202008',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False)

# Build the model using a pre-trained CNN
base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Add additional layers to the pre-trained model
x = base_model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
output_layer = keras.layers.Dense(len(train_data.class_indices), activation='softmax')(x)

# Compile the model
model = keras.Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10)

# Save the model
model.save('pill_identification_model.h5')

# Define the path to the folder of images
folder_path = 'D:\Documents\Coding\CAPSTONE\ChatGPT\pillbox_production_images_full_202008'

# Define the target size for the images
target_size = (150, 150)

# Create an empty list to store the preprocessed images
preprocessed_images = []

# Loop over each file in the folder
for file in os.listdir(folder_path):
    # Check that the file is an image file
    if file.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image and resize it to the target size
        img = image.load_img(os.path.join(folder_path, file), target_size=target_size)
        # Convert the image to a Numpy array and add it to the list of preprocessed images
        x = image.img_to_array(img)
        preprocessed_images.append(x)

# Convert the list of preprocessed images to a Numpy array
preprocessed_images = np.array(preprocessed_images)

# Normalize the pixel values to be between 0 and 1
preprocessed_images /= 255.0

prediction = model.predict(preprocessed_images)
prediction_index = np.argmax(prediction)

# Get the pill name from the file name
filenames = sorted(os.listdir(folder_path))
pill_name = os.path.basename(filenames[prediction_index]).split('_')[1].replace('.jpg', '')

print("This pill is:", pill_name)
