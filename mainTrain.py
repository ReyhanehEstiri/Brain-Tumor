import cv2
import os
from PIL import Image

image_directory='datasets/'

# os.listdir(image_directory+'no/')

no_tumor_images=os.listdir(image_directory+'no/')

# print(no_tumor_images)
# path = 'no0.jpg'
# print(path.split('.')[1])

for i ,image_name in enumerate(no_tumor_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image,'RGB')

