import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = keras.models.load_model('tsc/shape_classifier.h5')

# Load the image and preprocess it
img_path = 'tsc/Untitled-1.png'
img = image.load_img(img_path, target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Rescale pixel values to [0, 1]

# Use the model to make a prediction
prediction = model.predict(img_array)
class_names = ['Circle', 'Square', 'Triangle']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"The predicted class is: {predicted_class}")
print(f"The confidence in the prediction is: {confidence}")
