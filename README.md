# Binary Image Classification using CNN

# Project Overview
This project implements a Convolutional Neural Network (CNN) to perform binary image classification. The model classifies human face images into Male or Female categories using deep learning techniques.

# Objective
- Build a CNN model to automatically extract facial features
- Classify images into two classes: Male and Female
- Improve model generalization and reduce overfitting


# Dataset
- Dataset: Men vs Women (Kaggle)
- Classes: Male, Female
- Data Split:
  - Training: 80%
  - Testing: 20%


# Technologies Used
- Programming Language: Python
- Libraries & Frameworks:
  - TensorFlow
  - Keras
  - NumPy
  - OpenCV / PIL
- Model Type: Convolutional Neural Network (CNN)


# Data Preprocessing
- Resized images to 256 × 256 pixels
- Normalized pixel values to the range 0–1
- Converted images into tensors
- Split dataset into training and testing sets


# Model Architecture
- Convolutional layers with ReLU activation
- MaxPooling layers for spatial reduction
- Flatten layer to convert feature maps into vectors
- Fully connected Dense layers
- Dropout to prevent overfitting
- BatchNormalization to stabilize training
- Output layer with Sigmoid activation


# Model Training
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 10


# Results
- Achieved good accuracy on training and testing datasets
- Reduced overfitting using Dropout and BatchNormalization
- Successfully classified unseen face images

# Prediction Output
- Value close to 0 → Male
- Value close to 1 → Female


# Learning Outcomes
- CNN architecture design
- Image preprocessing techniques
- Overfitting control using Dropout and BatchNormalization
- Model evaluation and performance analysis
- Binary classification using Sigmoid activation

# Future Enhancements
- Extend to multiclass classification (age groups, emotions)
- Replace Sigmoid with Softmax
- Increase dataset size
- Deploy as a web application


# How to Run the Project
```bash
git clone https://github.com/your-username/binary-image-classification-cnn.git
pip install -r requirements.txt
python train.py
