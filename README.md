# Multi-Class Perceptron for Digit Recognition

## Overview
This project involves the implementation of a multi-class perceptron model using the one-vs-all strategy. The purpose of this model is to recognize and classify handwritten digits (0-9) from the MNIST dataset. The perceptron models are trained using a sigmoid activation function, adapting the basic perceptron model for multi-class classification.

## Features
Multi-Class Classification: Implements one-vs-all strategy to classify digits from 0 to 9.
Sigmoid Activation Function: Utilizes the sigmoid function for smooth, differentiable binary classification within each perceptron.
Digit Image Preprocessing: Includes image resizing and normalization for efficient training.

## Dataset
The MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits, is used for training and testing the model. The dataset is preprocessed by resizing the images to 20x20 pixels and normalizing the pixel values.

## Requirements
Python 3.x
NumPy
Pandas
OpenCV (cv2)

## Implementation Details
Neuron: Class representing a single neuron with forward pass capability.
Sigmoid: Sigmoid activation function for binary classification.
InputData: Class for loading and preprocessing the dataset.
Training and Testing: Classes for training the model using gradient descent and testing its accuracy.
The model is trained separately for each digit (0-9) using the one-vs-all strategy.

## Usage
Load the MNIST dataset.
Preprocess the data to 20x20 pixel images.
Initialize and train a perceptron model for each digit class.
Evaluate the model performance on a development/test set.

## Results
The accuracy of the model is calculated by comparing the aggregated predictions of each binary classifier against the actual labels in the development set.
The model's performance is dependent on the number of iterations and learning rate, which can be adjusted in the train method of the Training class.
