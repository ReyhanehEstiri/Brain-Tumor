# Brain Tumor Classification

## Overview

This repository contains two scripts related to brain tumor classification using Convolutional Neural Networks (CNNs). The first script trains a CNN model on MRI images to classify them into two categories: **tumor** and **no tumor**. The second script uses a pre-trained model to make predictions on new images.

## Requirements

To run these scripts, you need the following Python libraries:

- OpenCV (`cv2`)
- TensorFlow (`tensorflow`)
- Pillow (`PIL`)
- NumPy (`numpy`)
- scikit-learn (`sklearn`)

Install the required libraries using pip:

```bash
pip install opencv-python tensorflow pillow numpy scikit-learn
```

## Training Script

### Description

The training script prepares and trains a CNN model on a dataset of MRI images to classify them as either **tumor** or **no tumor**. The trained model is saved for future use.

### Data Preparation

1. **Dataset Structure:**
   Ensure your dataset is organized into two folders within a directory named `datasets/`:
   - `datasets/no/` for images without tumors
   - `datasets/yes/` for images with tumors

2. **Image Format:**
   Images should be in `.jpg` format.

### Usage

1. **Run the Script:**
   Execute the training script using Python:

   ```bash
   python main_Train.py
   ```

2. **Model File:**
   The trained model will be saved as `BrainTumor10Epochs.h5`.

### Code Overview

- **Data Loading and Preprocessing:**
  Images are resized to 64x64 pixels and normalized.
  
- **Model Architecture:**
  A CNN with three convolutional layers, max pooling, flattening, and dense layers.

- **Training:**
  The model is compiled with categorical crossentropy loss and trained for 10 epochs.

## Prediction Script

### Description

The prediction script uses a pre-trained CNN model to classify MRI images and determine whether a tumor is present.

### Usage

1. **Prepare the Script:**
   Place the pre-trained model file (`BrainTumor10Epochs.h5`) and the image to classify (`pred0.jpg`) in the correct locations.

2. **Run the Script:**
   Execute the prediction script using Python:

   ```bash
   python mainTest.py
   ```
3. **Output:**
   The script will print the prediction result to the console.

### Code Overview

- **Model Loading:**
  The pre-trained model is loaded from `BrainTumor10Epochs.h5`.

- **Image Processing:**
  The image is resized to 64x64 pixels, converted to a NumPy array, and expanded to fit the model’s input.

- **Prediction:**
  The image is passed through the model to get the prediction.



