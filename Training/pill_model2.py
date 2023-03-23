import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

#mnist = tf.keras.datasets.mnist

batch_size = 32
img_height = 150
img_width = 150

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0


data_dir = r"C:\Users\Arnab\iCloudDrive\Documents\Capstone\MedicationStation_Capstone\Medication_Dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



num_classes = 9

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])


model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)

model.save('pill_identification_model.h5')


folder_path = r"C:\Users\Arnab\iCloudDrive\Documents\Capstone\Test"

# Define the target size for the images
target_size = (150, 150)

# Create an empty list to store the preprocessed images
preprocessed_images = []

# Loop over each file in the folder
for file in os.listdir(folder_path):
    # Check that the file is an image file
    if file.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image and resize it to the target size
        img = tf.keras.utils.load_img(os.path.join(folder_path, file), target_size=target_size)
        # Convert the image to a Numpy array and add it to the list of preprocessed images
        x = tf.keras.utils.img_to_array(img)
        preprocessed_images.append(x)

# Convert the list of preprocessed images to a Numpy array
preprocessed_images = np.array(preprocessed_images)

# Normalize the pixel values to be between 0 and 1
preprocessed_images /= 255.0

prediction = model.predict(preprocessed_images)
print(prediction)

prediction_index = np.argmax(prediction)

# Get the pill name from the file name
filenames = sorted(os.listdir(folder_path))
pill_name = os.path.basename(filenames[prediction_index]).split('_')[1].replace('.jpg', '')
# filename = filenames[0]
# pill_name = filename.split('.')[0]  # remove the file extension
# pill_name = ''.join([i for i in pill_name if not i.isdigit()])  # remove the number

print("This pill is:", pill_name)

