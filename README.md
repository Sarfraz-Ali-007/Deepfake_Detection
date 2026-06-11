# Deepfake Detection System

A Deep Learning-based Deepfake Detection System built using TensorFlow and MobileNetV2. This project classifies facial images as **Real** or **Fake** using transfer learning and computer vision techniques.

---

## Overview

Deepfakes are synthetic media generated using AI techniques that can manipulate visual content and create realistic but fake images or videos. Detecting such content is becoming increasingly important for digital security, media verification, and misinformation prevention.

This project uses a pretrained MobileNetV2 model to learn visual patterns that distinguish authentic images from AI-generated or manipulated images.

---

## Key Features

- Binary classification: Real vs Fake
- Transfer Learning with MobileNetV2
- Image preprocessing and augmentation
- Model evaluation on unseen test data
- Streamlit-based prediction interface
- End-to-end Deep Learning pipeline

---

## Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Model Architecture | MobileNetV2 |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Deployment | Streamlit |

---

## Dataset Structure

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

The dataset is divided into:

- Training Set
- Validation Set
- Test Set

This split helps evaluate model performance on unseen data and reduce overfitting.

---

## Data Preprocessing

Before training, all images were normalized:

```python
rescale = 1./255
```

Pixel values were scaled from:

```text
0–255 → 0–1
```

---

## Data Augmentation

To improve generalization and reduce overfitting, the following augmentation techniques were applied:

```python
rotation_range=20
zoom_range=0.2
horizontal_flip=True
```

Benefits:

- Increases dataset diversity
- Improves robustness
- Reduces overfitting

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Classification Type | Binary |
| Image Normalization | 1/255 |
| Rotation Range | 20° |
| Zoom Range | 0.2 |
| Horizontal Flip | Enabled |

---

## Model Architecture

This project uses **MobileNetV2** as the feature extractor through transfer learning.

### Workflow

1. Load pretrained MobileNetV2 weights
2. Remove the original classification layer
3. Add custom dense layers
4. Train on deepfake dataset
5. Evaluate on validation and test sets
6. Deploy for real-time prediction

### Why MobileNetV2?

- Lightweight architecture
- Fast inference
- Efficient transfer learning
- Strong image classification performance

---

## Project Structure

```text
Deepfake_Detection/
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
│
├── screenshots/
│   ├── home.png
│   ├── prediction_real.png
│   └── prediction_fake.png
│
└── dataset/
```

---

## Running the Project

### Clone Repository

```bash
git clone https://github.com/Sarfraz-Ali-007/Deepfake_Detection.git
cd Deepfake_Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Predictions

```bash
python predict.py
```

### Launch Streamlit Application

```bash
streamlit run app.py
```

---

## Screenshots

### Application Home Page

Add:

```text
screenshots/home.png
```

### Real Image Prediction

Add:

```text
screenshots/prediction_real.png
```

### Fake Image Prediction

Add:

```text
screenshots/prediction_fake.png
```

---

## Future Enhancements

- Video Deepfake Detection
- Real-Time Webcam Detection
- Explainable AI Visualizations
- Model Quantization and Optimization
- Multi-Class Deepfake Classification

---

## Skills Demonstrated

This project demonstrates practical experience in:

- Deep Learning
- Computer Vision
- Transfer Learning
- TensorFlow
- Keras
- MobileNetV2
- Data Augmentation
- Image Classification
- Model Evaluation
- Streamlit Deployment
- Python Development

---

## Author

### Sarfraz Ali

GitHub:
https://github.com/Sarfraz-Ali-007

LinkedIn:
https://www.linkedin.com/in/sarfraz-ali-ai/

---
