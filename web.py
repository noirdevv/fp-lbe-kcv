import os
import tempfile
import joblib
import numpy as np
from PIL import Image
import streamlit as st

# Tambahan import untuk PyTorch
import torch
from torchvision import models, transforms
import torch.nn as nn

st.set_page_config(
    page_title="Website Rating Predictor", page_icon="🌐", layout="centered"
)

st.title("🌐 Website Visual Rating Predictor")
st.write(
    "Upload a screenshot of any website to generate its predicted visual appeal ratings."
)

# Define the local path to your saved model file here
MODEL_PATH = "model_knn.joblib"

@st.cache_resource
def load_injected_model(path: str):
    """Loads the developer-injected model from disk once and caches it."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_cnn_extractor():
    """Loads a pre-trained CNN model (e.g., ResNet50) as a feature extractor."""
    # Menggunakan ResNet50 pre-trained
    weights = models.ResNet50_Weights.DEFAULT
    cnn_model = models.resnet50(weights=weights)
    
    # Mengganti layer klasifikasi terakhir agar model mengembalikan vektor fitur (bukan probabilitas kelas)
    cnn_model.fc = nn.Identity()
    cnn_model.eval() # Set mode evaluasi
    
    # Mengambil transformer standar dari pre-trained weights
    preprocess_transform = weights.transforms()
    
    return cnn_model, preprocess_transform

# Load kedua model
model = load_injected_model(MODEL_PATH)
feature_extractor, preprocess = load_cnn_extractor()

# Fungsi ekstraksi fitur yang Anda berikan
def extract_cnn_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_batch = img_tensor.unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(img_batch)

    feature_vector = features.squeeze().numpy()
    return feature_vector

uploaded_file = st.file_uploader(
    "Upload a website screenshot (.png, .jpg, .jpeg):",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    preview_image = Image.open(uploaded_file)
    st.image(
        preview_image, caption="Uploaded Website Preview", use_container_width=True
    )

    if st.button("Predict Ratings", type="primary"):
        if model is None:
            st.error("The prediction service is currently unavailable. Model missing.")
        else:
            # Save the uploaded file to a temporary file path
            suffix = os.path.splitext(uploaded_file.name)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            try:
                with st.spinner("Extracting features and analyzing website..."):
                    # 1. Ekstrak fitur gambar menggunakan PyTorch CNN
                    feature_vector = extract_cnn_features(temp_file_path)
                    
                    # 2. Reshape vektor menjadi 2D array (1 baris, N kolom) karena scikit-learn membutuhkan format ini
                    feature_vector_2d = feature_vector.reshape(1, -1)
                    
                    # 3. Prediksi rating menggunakan model .joblib Anda
                    prediction = model.predict(feature_vector_2d)

                st.success("Analysis complete!")
                st.subheader("Predicted Human Ratings")

                # Flatten output to handle 1D or 2D results
                pred_flat = np.ravel(prediction)

                if len(pred_flat) == 1:
                    st.metric(
                        label="Predicted Rating Score",
                        value=f"{pred_flat[0]:.2f}",
                    )
                elif len(pred_flat) == 2:
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Response X", f"{pred_flat[0]:.2f}")
                    col2.metric("Predicted Response Y", f"{pred_flat[1]:.2f}")
                else:
                    st.write("Predicted Values:", pred_flat)

            except Exception as err:
                st.error(f"Prediction failed: {err}")

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)