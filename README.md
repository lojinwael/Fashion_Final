# Fashion_Final

# Fashion MNIST Neural Network Classification

This project implements and compares multiple Deep Neural Network (DNN) models using PyTorch for classifying the Fashion MNIST dataset.

## Project Overview

The goal of this project is to build, train, and evaluate different neural network architectures for image classification on the Fashion MNIST dataset.  
Three different models were tested with different architectures, activation functions, dropout rates, and optimization strategies.

The project includes:

- Data preprocessing and normalization
- Training/validation/test split
- Multiple neural network architectures
- Batch Normalization and Dropout
- Learning rate scheduling
- Early stopping
- Performance visualization
- Model comparison

---

## Dataset

The project uses the Fashion MNIST dataset from TensorFlow/Keras.

Dataset details:

- 60,000 grayscale images
- Image size: 28×28 pixels
- 10 clothing categories

Examples of classes:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Official dataset: https://github.com/zalandoresearch/fashion-mnist

---

## Technologies Used

- Python
- PyTorch
- TensorFlow/Keras
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas

---

# Data Preprocessing

The following preprocessing steps were applied:

1. Loaded Fashion MNIST dataset
2. Flattened images from 28×28 into 784 features
3. Split data into:
   - Training set
   - Validation set
   - Test set
4. Applied feature scaling using `StandardScaler`
5. Converted data into PyTorch tensors
6. Created DataLoaders for batch training

---

# Model Architectures

## Model 1

Architecture:

- Linear(784 → 256)
- BatchNorm1d
- ReLU
- Dropout(0.3)

- Linear(256 → 128)
- BatchNorm1d
- ReLU
- Dropout(0.3)

- Linear(128 → 10)

### Features

- Adam optimizer
- Learning rate scheduler (`StepLR`)
- Early stopping
- CrossEntropyLoss

### Final Results

- Final Test Accuracy: **89.31%**

---

## Model 2

Architecture:

- Linear(784 → 256)
- BatchNorm1d
- Tanh
- Dropout(0.2)

- Linear(256 → 128)
- BatchNorm1d
- Tanh
- Dropout(0.2)

- Linear(128 → 64)
- BatchNorm1d
- Tanh
- Dropout(0.2)

- Linear(64 → 10)

### Features

- Adam optimizer
- Learning rate scheduler (`StepLR`)
- CrossEntropyLoss

### Final Results

- Final Test Accuracy: **88.71%**

---

## Model 3 (Best Model)

Architecture:

- Linear(784 → 512)
- BatchNorm1d
- ReLU
- Dropout(0.3)

- Linear(512 → 256)
- BatchNorm1d
- ReLU
- Dropout(0.3)

- Linear(256 → 128)
- BatchNorm1d
- ReLU

- Linear(128 → 10)

### Features

- Adam optimizer
- Lower learning rate (`0.0001`)
- CrossEntropyLoss

### Final Results

- Final Test Accuracy: **89.44%**

---

# Final Comparison

| Model | Test Accuracy |
|---|---|
| Model 1 | 89.31% |
| Model 2 | 88.71% |
| Model 3 | 89.44% |

### Best Performing Model

Model 3 achieved the highest accuracy with **89.44%**.

---

# Training Techniques Used

## Batch Normalization

Used to:
- Stabilize training
- Improve convergence speed
- Reduce internal covariate shift

## Dropout

Used to:
- Prevent overfitting
- Improve model generalization

## Early Stopping

Implemented in Model 1 to stop training when validation loss stopped improving.

## Learning Rate Scheduler

Used to gradually reduce learning rate during training for better optimization.

---

# Visualizations

The project includes:

- Training Loss Curve
- Validation Loss Curve
- Training Accuracy Curve
- Validation Accuracy Curve

These visualizations help analyze:
- Model convergence
- Overfitting
- Generalization performance

---

# How to Run the Project

## Install Dependencies

```bash
pip install torch torchvision tensorflow scikit-learn matplotlib seaborn pandas
