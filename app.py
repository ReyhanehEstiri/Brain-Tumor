import cv2
from tensorflow import keras
from PIL import Image
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def main():
    model = keras.models.load_model('BrainTumor50EpochsCategorical.h5')

    Tk().withdraw()
    file_path = askopenfilename(title="Select Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        print("No file selected. Exiting...")
        return


    image = cv2.imread(file_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))

    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)

    result = model.predict(input_img)
    # print(result)

    # Convert prediction result to binary (0 or 1)
    predicted_class = np.argmax(result, axis=1)[0]


    if predicted_class == 0:
        print("No brain tumor")
        print(predicted_class)
        cv2.imshow("Selected Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif predicted_class == 1:
        print("Yes, brain tumor")
        print(predicted_class)
        cv2.imshow("Selected Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(predicted_class)

if __name__ == "__main__":
    main()
