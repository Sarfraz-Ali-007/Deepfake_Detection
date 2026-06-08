import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="centered"
)

# ---------------- CONFIG ----------------
IMG_SIZE = 224

# Your HF model repository
REPO_ID = "freemldl/Deepfake_Detection"
MODEL_FILE = "model.weights.h5"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():

    # Download model from HF Hub
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        repo_type="space"
    )

    base_model = MobileNetV2(
        weights=None,
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

    model.load_weights(model_path)

    return model

# Load model
model = load_model()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ---------------- PREDICTION ----------------
def predict_image(image):

    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0][0]

    return prediction

# ---------------- UI ----------------
st.title("🔍 Deepfake Detection")

st.write(
    "Upload an image and the model will predict whether it is REAL or FAKE."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Analyzing..."):

            score = predict_image(image)

            # score >= 0.5 = REAL
            # score < 0.5 = FAKE

            if score >= 0.5:
                st.success(
                    f"REAL Image\n\nConfidence: {score:.2%}"
                )
            else:
                st.error(
                    f"FAKE Image\n\nConfidence: {(1-score):.2%}"
                )

            st.write(f"Raw Model Score: {score:.6f}")
