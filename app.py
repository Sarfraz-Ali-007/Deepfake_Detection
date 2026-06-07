import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
IMG_SIZE = 128

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="freemldl/Deepfake_Detection",  
        filename="fake_face_detector.h5"
    )

    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Deepfake Face Detector",
    page_icon="🧠",
    layout="centered"
)

# ---------------- UI ----------------
st.title("🧠 Deepfake Face Detector")

st.write(
    "Upload a face image and the model will predict whether it is REAL or FAKE."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Detect"):

        # Preprocess image
        img = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img)

        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )

        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = float(model.predict(img_array, verbose=0)[0][0])

        st.divider()

        if pred > 0.5:

            confidence = pred * 100

            st.error(
                f"❌ FAKE FACE DETECTED\n\nConfidence: {confidence:.2f}%"
            )

            st.progress(min(int(confidence), 100))

        else:

            confidence = (1 - pred) * 100

            st.success(
                f"✅ REAL FACE DETECTED\n\nConfidence: {confidence:.2f}%"
            )

            st.progress(min(int(confidence), 100))

        st.write(f"Raw Prediction Score: `{pred:.4f}`")
