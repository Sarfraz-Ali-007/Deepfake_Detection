# Deepfake_Detection

# Deepfake Detection System

A Deep Learning-based Deepfake Detection System that identifies whether an image is **Real** or **Fake** using TensorFlow and MobileNetV2. The project leverages transfer learning to achieve efficient and accurate image classification and provides an easy-to-use interface for testing images.

---

## Project Overview

Deepfakes are AI-generated images or videos that can manipulate visual content and spread misinformation. This project aims to detect deepfake images using a Convolutional Neural Network (CNN) based on MobileNetV2.

The model is trained on labeled real and fake face images and can classify uploaded images in real time.

---

## Features

- Detects Real and Fake Images
- Transfer Learning with MobileNetV2
- Binary Image Classification
- Data Augmentation
- Model Evaluation on Validation and Test Data
- Real-Time Prediction
- Easy Deployment with Streamlit

---

## Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Streamlit

---

## Dataset

The dataset contains two classes:

- Real Images
- Fake Images

Dataset Structure:

```text
dataset/
│
├── train/
│   ├── real/
│   └── fake/
│
├── valid/
│   ├── real/
│   └── fake/
│
└── test/
    ├── real/
    └── fake/
```

---

## Model Architecture

The project uses Transfer Learning with MobileNetV2.

Steps:

1. Load pretrained MobileNetV2 weights
2. Freeze base layers
3. Add custom classification layers
4. Train on Deepfake dataset
5. Evaluate performance on validation and test sets

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Image Size | 160 × 160 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 5 |

---

## Project Structure

```text
Deepfake_Detection/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── model/
│   └── deepfake_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── screenshots/
│   ├── home.png
│   ├── real_prediction.png
│   └── fake_prediction.png
│
└── dataset/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sarfraz-Ali-007/Deepfake_Detection.git
```

Move into the project folder:

```bash
cd Deepfake_Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python train.py
```

---

## Run Prediction

```bash
python predict.py
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

## Screenshots

### Home Page

Add screenshot here:

```text
screenshots/home.png
```

### Real Image Prediction

Add screenshot here:

```text
screenshots/real_prediction.png
```

### Fake Image Prediction

Add screenshot here:

```text
screenshots/fake_prediction.png
```

---

## Future Improvements

- Video Deepfake Detection
- Explainable AI Visualizations
- Higher Accuracy Models
- Real-Time Webcam Detection

---

## Skills Demonstrated

- Deep Learning
- Computer Vision
- TensorFlow
- CNN
- Transfer Learning
- Data Preprocessing
- Model Evaluation
- Streamlit Deployment
- Python Development

---

## Author

**Sarfraz Ali**

GitHub:
https://github.com/Sarfraz-Ali-007

LinkedIn:
(Add your LinkedIn profile link)

---
