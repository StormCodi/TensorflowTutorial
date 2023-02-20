import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps

# Load the trained model from file
model = keras.models.load_model('char/char_classifier.h5')

# Load the image to predict
img = Image.open('char/Untitled-1.png')

# Convert the image to grayscale and resize it
img = img.convert('L')
img = img.resize((28, 28))

# Invert the image colors (black background, white characters)
img = ImageOps.invert(img)

# Preprocess the image to match the training data
img_arr = np.array(img) / 255.0
img_arr = img_arr.reshape((1, 28, 28, 1))

# Make the prediction
prediction = model.predict(img_arr)
class_index = np.argmax(prediction)

# Print the predicted class
print(f"Predicted class: {class_index}")
