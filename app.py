import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

# ---------------- CONFIG ----------------
IMG_SIZE = 128
REPO_ID = "freemldl/Deepfake_Detection"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():

    files = list_repo_files(REPO_ID)

    h5_files = [f for f in files if f.endswith(".h5")]

    if not h5_files:
        raise Exception(
            f"No .h5 file found in Hugging Face repo.\nFiles found: {files}"
        )

    model_file = h5_files[0]

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=model_file
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

        img = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img)

        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            img_array
        )

        img_array = np.expand_dims(img_array, axis=0)

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
