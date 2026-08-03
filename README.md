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
