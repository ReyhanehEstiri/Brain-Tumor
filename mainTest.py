import cv2
from tensorflow import keras
# from keras.models import load_model
from PIL import Image
import numpy as np

# model = keras.models.load_model('BrainTumor10EpochsCategorical.h5')
model = keras.models.load_model('BrainTumor10Epochs.h5')

image = cv2.imread('pred\\pred0.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))

img = np.array(img)
input_img = np.expand_dims(img,axis=0)

# print(img)
result = model.predict(input_img)
print(result)