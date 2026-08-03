# CVI620 Self-Driving Car

## Project Overview

This project was developed as the final project for **CVI620 – Computer Vision**.

The goal of the project is to build an end-to-end self-driving car system that predicts a vehicle's steering angle directly from camera images. The project follows the NVIDIA end-to-end autonomous driving approach and uses a Convolutional Neural Network (CNN) implemented with TensorFlow/Keras.

The system is designed to process images captured from a driving simulator, learn the relationship between the road scene and steering input, and then use the trained model to predict steering commands for unseen images.

## Project Objectives

The main objectives of the project are:

* Collect and organize driving images with corresponding steering-angle values.
* Analyze the distribution of steering data.
* Prepare driving images for neural-network training.
* Apply data augmentation to improve dataset diversity.
* Build an NVIDIA-style CNN for steering-angle regression.
* Train and validate the CNN using collected driving data.
* Save the best-performing model for later use.
* Load the trained model and perform steering-angle inference.
* Integrate the prediction pipeline with the self-driving simulation workflow.

## Our Approach

The project uses an **end-to-end deep-learning approach**. Instead of manually detecting road lanes and calculating steering commands using predefined rules, the CNN learns the relationship between camera images and steering angles directly from training examples.

The overall workflow is:

**Driving Data → Data Analysis → Image Preprocessing/Augmentation → CNN Model → Training → Saved Model → Steering Prediction**

### 1. Data Collection and Analysis

Driving data consists of road images together with steering-angle values stored in CSV driving logs.

A histogram notebook is included in the project to examine the steering-angle distribution of the training and testing datasets. This helps identify imbalances in the collected driving data.

### 2. Image Preprocessing

Images are prepared for the NVIDIA CNN using a standardized input resolution of:

`66 × 200 × 3`

The preprocessing pipeline focuses the model on useful road information and ensures that all images have a consistent format before they are passed to the neural network.

### 3. Data Augmentation

Data augmentation is used to increase the diversity of the training data and reduce overfitting.

The project experiments with transformations including:

* Horizontal image flipping with steering-angle correction
* Brightness adjustment
* Random cropping and zooming
* Image rotation

These transformations simulate different visual and driving conditions while allowing the model to learn more robust steering behaviour.

### 4. CNN Steering Model

The steering model follows the NVIDIA end-to-end CNN architecture.

The network contains multiple convolutional layers for extracting visual road features, followed by fully connected layers that estimate the appropriate steering value.

Because steering prediction is a regression problem, the final layer contains a single output representing the predicted steering angle.

### 5. Training

The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss.

The training pipeline also supports:

* Validation monitoring
* Early stopping
* Model checkpointing
* Saving the best model based on validation loss
* Saving training history for analysis

### 6. Inference

After training, the saved Keras model can be loaded by the inference pipeline.

A camera image is passed into the model and the CNN returns a single steering-angle prediction. This prediction can then be used as part of the autonomous-driving simulation workflow.

## Technologies

The project is primarily built with:

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Udacity Self-Driving Car Simulator

## Course

**CVI620 – Computer Vision**

Final Project: End-to-End Self-Driving Car

## Installation and Setup

### Prerequisites

Before running the project, make sure the following are installed:

* Python 3
* pip
* Git
* A Python virtual environment is recommended

### 1. Clone the Repository

```bash
git clone https://github.com/Mohammad-prs/CVI620-Self-Driving-Car.git
cd CVI620-Self-Driving-Car
```

### 2. Create a Virtual Environment

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

The project uses several Python libraries for image processing, numerical operations, model development, and simulator interaction.

Main dependencies include:

* NumPy – numerical operations and image arrays
* OpenCV – image loading, preprocessing, and transformations
* TensorFlow/Keras – CNN model creation, training, and inference
* Pandas – driving-log and dataset processing
* Matplotlib – data visualization and steering-distribution analysis
* PyAutoGUI – interaction with the simulator during inference

### Dataset Setup

The project uses driving images together with CSV driving logs containing image paths and steering values.

Training and testing data should be stored in their corresponding project data directories. Each CSV entry associates an image with its steering-angle target.

Before training, verify that the image paths referenced by the driving logs point to the correct image directories.

### Model Output

During training, the project stores generated model artifacts inside the configured models directory.

The best model is saved as:

```text
best_model.keras
```

Training history is also saved for later analysis.

## Model Architecture

The project uses a CNN inspired by NVIDIA's end-to-end learning architecture for autonomous driving.

The model accepts a preprocessed road image with the input shape:

```text
66 × 200 × 3
```

The CNN uses a sequence of convolutional layers to extract visual features from the road image.

The convolutional feature extractor is followed by fully connected layers that gradually reduce the learned representation to a single output.

The final output represents the predicted steering angle.

### Model Pipeline

```text
Camera Image
     ↓
Image Preprocessing
     ↓
66 × 200 × 3 Image
     ↓
Convolutional Layers
     ↓
Feature Extraction
     ↓
Fully Connected Layers
     ↓
Steering Angle
```

## Image Preprocessing

Before an image is passed to the neural network, preprocessing is used to produce a consistent model input.

The preprocessing workflow includes:

1. Cropping the image to focus primarily on the useful road region.
2. Converting the image to the YUV colour space.
3. Applying Gaussian blur.
4. Resizing the image to `200 × 66`.
5. Normalizing pixel values.

The final processed image has the shape:

```text
(66, 200, 3)
```

## Data Augmentation

The project also experiments with data augmentation to increase variation in the training dataset.

Augmentation operations include:

### Horizontal Flip

Images may be flipped horizontally. When an image is flipped, the steering angle is multiplied by `-1` so that the label remains consistent with the transformed driving direction.

### Brightness Adjustment

Random brightness changes simulate different lighting conditions.

### Random Crop and Zoom

Random cropping followed by resizing introduces variation in camera framing and road position.

### Rotation

Small random rotations introduce additional visual variation into the training data.

## Training Configuration

The model is designed as a regression model because the output is a continuous steering value.

Training uses:

* Adam optimizer
* Mean Squared Error (MSE) loss
* Mean Absolute Error (MAE) metric
* Validation monitoring
* Early stopping
* Model checkpointing

Early stopping helps prevent unnecessary training once validation performance stops improving.

Model checkpointing preserves the model with the best validation loss.

## Training Outputs

The training process can generate:

```text
models/best_model.keras
models/training_history.json
```

The saved Keras model is used later by the inference pipeline, while the training history can be used to analyze model performance.

## Project Structure

The project separates configuration, model definition, training, inference, utilities, dataset analysis, and driving data.

```text
CVI620-Self-Driving-Car/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── traindata/
│   ├── driving_log.csv
│   └── IMG/
│
├── testdata/
│   ├── driving_log.csv
│   └── IMG/
│
└── histogram.ipynb
```

### Main Source Files

#### `config.py`

Contains shared configuration values used by the project, including model input dimensions, training parameters, paths, and other project settings.

#### `model.py`

Defines the NVIDIA-style convolutional neural network used to predict steering values from road images.

#### `train.py`

Contains the model-training workflow, including model creation, compilation, callbacks, checkpointing, and training-history handling.

#### `inference.py`

Loads a saved trained model and performs steering-angle prediction.

#### `utils.py`

Contains reusable helper functionality used by the project.

#### `histogram.ipynb`

Provides dataset analysis and experimentation, including steering-angle distribution analysis, image augmentation, preprocessing, and batch-generation work.

## Running the Project

### Model Training

The training workflow is located in:

```text
src/train.py
```

Before running training, make sure that the dataset paths and generators are correctly connected to the training and validation data.

The training script is responsible for:

* Building the CNN
* Compiling the model
* Configuring callbacks
* Training the model
* Monitoring validation performance
* Saving the best model
* Saving training history

### Inference

Inference is handled through:

```text
src/inference.py
```

The inference workflow loads the saved Keras model and uses it to generate steering predictions.

Conceptually, the inference process is:

```text
Road Image
    ↓
Preprocessing
    ↓
Trained CNN
    ↓
Predicted Steering Angle
```

### Dataset Analysis

The histogram notebook can be used to inspect the distribution of steering values in the collected training and testing data.

Analyzing this distribution is useful because an autonomous-driving dataset may contain many straight-driving examples compared with left and right turns.

The augmentation experiments can then be used to introduce additional variation into the training dataset.

## Reproducibility

For consistent experiments, the project configuration defines a random seed.

Keeping model parameters, paths, and training settings in the configuration module also makes experiments easier to reproduce and modify.

## Final Demonstration

The final demonstration presents the complete workflow of the self-driving car project, from the source code and dataset to steering-angle prediction.

### Demo Flow

The demonstration should cover the following components:

1. **Project Repository**

   Show the GitHub repository and briefly explain the organization of the project.

2. **Driving Dataset**

   Show the training images and `driving_log.csv` data.

   Explain that each driving image is associated with a steering value used as the target during supervised learning.

3. **Dataset Analysis**

   Open the histogram notebook and show the steering-angle distribution.

   Briefly explain why analyzing the distribution is important before training.

4. **Image Preprocessing**

   Demonstrate or explain the preprocessing pipeline:

   ```text
   Original Image
         ↓
   Crop Road Region
         ↓
   Convert to YUV
         ↓
   Gaussian Blur
         ↓
   Resize to 200 × 66
         ↓
   Normalize
   ```

5. **Data Augmentation**

   Explain the augmentation techniques used to increase training-data diversity:

   * Horizontal flipping
   * Brightness changes
   * Cropping/zooming
   * Rotation

6. **CNN Architecture**

   Open `model.py` and explain that the NVIDIA-style CNN extracts visual road features and predicts a continuous steering value.

7. **Training Pipeline**

   Open `train.py` and show:

   * Model creation
   * Adam optimizer
   * MSE loss
   * Validation monitoring
   * Early stopping
   * Model checkpointing

8. **Inference**

   Open `inference.py` and demonstrate how the trained model is loaded and used to produce steering predictions.

9. **Final Result**

   Show the final available autonomous-driving or steering-prediction result and explain how camera input is transformed into a steering command.

## Demo Summary

The project demonstrates an end-to-end computer-vision pipeline where driving images are processed by a convolutional neural network to estimate vehicle steering.

The main stages are:

```text
Data Collection
      ↓
Data Analysis
      ↓
Preprocessing & Augmentation
      ↓
CNN Training
      ↓
Model Validation
      ↓
Model Saving
      ↓
Inference
      ↓
Steering Prediction
```

This approach demonstrates how deep learning can learn driving behaviour directly from visual road information instead of relying entirely on manually designed lane-detection and steering rules.
