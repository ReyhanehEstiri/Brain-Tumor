import cv2
from tensorflow import keras
from PIL import Image
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def main():
    # Load the pre-trained model
    model = keras.models.load_model('BrainTumor10EpochsCategorical.h5')

    # Open a file dialog to select an image file
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_path = askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        print("No file selected. Exiting...")
        return

    # Load and preprocess the image
    image = cv2.imread(file_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))

    # Convert the image to a numpy array and expand dimensions to match model input
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)

    # Predict using the model
    result = model.predict(input_img)
    # print(result)

    # Convert prediction result to binary (0 or 1)
    predicted_class = np.argmax(result, axis=1)[0]



    # Output the result
    if predicted_class == 0:
        print("No brain tumor")
        print(predicted_class)
    elif predicted_class == 1:
        print("Yes, brain tumor")
        print(predicted_class)
    else:
        print(predicted_class)

if __name__ == "__main__":
    main()
