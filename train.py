import os, json

kaggle_config = {
    "username": "your username",
    "key": "your key"
}

# Create .kaggle folder
os.makedirs("/root/.kaggle", exist_ok=True)

# Save kaggle.json
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump(kaggle_config, f)

# Set permissions
os.chmod("/root/.kaggle/kaggle.json", 600)

print("kaggle.json created successfully!")

# ================================
# DOWNLOAD + EXTRACT YOUR DATASET
# ================================

# Replace with your dataset path:
DATASET = "xhlulu/140k-real-and-fake-faces"

# Download
!kaggle datasets download -d $DATASET

# Unzip
import zipfile, os

zip_file = DATASET.split("/")[-1] + ".zip"
extract_path = "/content/deepfake_dataset"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted to:", extract_path)

# ================================
# CHECK FOLDER STRUCTURE
# ================================
for root, dirs, files in os.walk(extract_path):
    print(f"{root} | {len(files)} files")

import os

dataset_path = "/content/deepfake_dataset"

print(os.listdir(dataset_path))

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

train_df = pd.read_csv("/content/deepfake_dataset/train.csv")
valid_df = pd.read_csv("/content/deepfake_dataset/valid.csv")
test_df = pd.read_csv("/content/deepfake_dataset/test.csv")

print(train_df.head())

print(train_df.columns)

print(valid_df.head())
print(valid_df.columns)

dataset_path = "/content/deepfake_dataset/real_vs_fake/real-vs-fake"

train_df['path'] = dataset_path + "/" + train_df['path']
valid_df['path'] = dataset_path + "/" + valid_df['path']
test_df['path'] = dataset_path + "/" + test_df['path']

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label_str',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='path',
    y_col='label_str',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = valid_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label_str',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=5
)
