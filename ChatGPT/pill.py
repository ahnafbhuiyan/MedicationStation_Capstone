# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Create a data generator for our dataset
datagen = ImageDataGenerator(rescale = 1./255)

# Use the data generator to read in our dataset
train_data = datagen.flow_from_directory(
    'D:\Documents\Coding\CAPSTONE\ChatGPT\pillbox_production_images_full_202008',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

print(len(train_data))
# Build the model using a pre-trained CNN
model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Add additional layers to the pre-trained model
x = model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(len(train_data.class_indices), activation='softmax')(x)

# Compile the model
model = keras.Model(inputs=model.input, outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10)

# Save the model
model.save('pill_identification_model.h5')

# Use the trained model to predict the name of a pill
from keras.preprocessing import image
import numpy as np

img = tf.keras.utils.load_img('D:\Documents\Coding\CAPSTONE\ChatGPT\pillbox_production_images_full_202008\Clonazepam_Tablet\\0.5_mg_Clonazepam_Tablet.jpg', target_size=(150, 150))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
print(train_data)
print(np.argmax(prediction))
pill_name = train_data.class_indices['Clonazepam_Tablet']

for name in train_data.class_indices:
    if train_data.class_indices[name] == np.argmax(prediction):
        print (name)
print("This pill is:", pill_name)


def findPillName(dict, index):
    for name in dict:
        if dict[name] == index:
            return name