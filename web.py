import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial import distance
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Website Rating Predictor", page_icon="🌐", layout="centered"
)

st.title("🌐 SeraSi: Sistem Rating Website")
st.write("Upload a screenshot of any website to generate its predicted visual appeal ratings.")

@st.cache_resource
def load_pipeline_artifacts():
    """Memuat model PCA, Scaler, dan ML final dari folder artifacts."""
    pca = joblib.load('models/pca_transformer.pkl')
    scaler = joblib.load('models/hybrid_scaler.pkl')
    ml_model = joblib.load('models/ml_model.pkl')
    return pca, scaler, ml_model

@st.cache_resource
def load_resnet_extractor():
    """Memuat ResNet18 dan memotong layer terakhir untuk ekstraksi fitur laten."""
    resnet = models.resnet18(pretrained=True)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.Resize((192, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return feature_extractor, transform

pca, scaler, ml_model = load_pipeline_artifacts()
cnn_extractor, cnn_transform = load_resnet_extractor()

def calculate_hasler_susstrunk_colorfulness(image_path):
    img = cv2.imread(image_path)
    B, G, R = cv2.split(img.astype(float))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
    return std_root + (0.3 * mean_root)

def extract_w3c_colors(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    w3c_palette = {
        'black': [0, 0, 0], 'silver': [192, 192, 192], 'gray': [128, 128, 128],
        'white': [255, 255, 255], 'maroon': [128, 0, 0], 'red': [255, 0, 0],
        'purple': [128, 0, 128], 'fuchsia': [255, 0, 255], 'green': [0, 128, 0],
        'lime': [0, 255, 0], 'olive': [128, 128, 0], 'yellow': [255, 255, 0],
        'navy': [0, 0, 128], 'blue': [0, 0, 255], 'teal': [0, 128, 128], 'aqua': [0, 255, 255]
    }
    palette_names = list(w3c_palette.keys())
    palette_values = np.array(list(w3c_palette.values()))
    dists = distance.cdist(pixels, palette_values, metric='euclidean')
    closest_color_indices = np.argmin(dists, axis=1)
    total_pixels = pixels.shape[0]
    
    color_percentages = {}
    for i, name in enumerate(palette_names):
        count = np.sum(closest_color_indices == i)
        color_percentages[name] = count / total_pixels
    return color_percentages

def calculate_horizontal_symmetry_and_balance(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    left_half = img[:, :w//2]
    right_half = img[:, w//2:]
    right_flipped = cv2.flip(right_half, 1)
    hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
    hist_right = cv2.calcHist([right_flipped], [0], None, [256], [0, 256])
    cv2.normalize(hist_left, hist_left, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_right, hist_right, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    symmetry = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_INTERSECT)
    M = cv2.moments(img)
    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2
    center_x = w / 2
    balance = 1.0 - (abs(cX - center_x) / center_x)
    return symmetry, balance

def quadtree_decomposition(img, min_size=16, var_threshold=100):
    h, w = img.shape
    if h <= min_size or w <= min_size:
        return 1
    variance = np.var(img)
    if variance > var_threshold:
        h2, w2 = h//2, w//2
        return (quadtree_decomposition(img[:h2, :w2], min_size, var_threshold) +
                quadtree_decomposition(img[:h2, w2:], min_size, var_threshold) +
                quadtree_decomposition(img[h2:, :w2], min_size, var_threshold) +
                quadtree_decomposition(img[h2:, w2:], min_size, var_threshold))
    return 1

def get_quadtree_leaves(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return quadtree_decomposition(img_gray)

def extract_layout_areas(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    total_area_screen = h * w
    edges = cv2.Canny(img, 100, 200)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_area_total = 0
    image_area_total = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        box_area = cw * ch
        if box_area == 0: continue
        box_edges = edges[y:y+ch, x:x+cw]
        edge_density = np.sum(box_edges > 0) / box_area
        if edge_density > 0.15:
            text_area_total += box_area
        else:
            image_area_total += box_area
    return text_area_total / total_area_screen, image_area_total / total_area_screen


def extract_cnn_features(image_path):
    img_pil = Image.open(image_path).convert('RGB')
    tensor = cnn_transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        cnn_raw = cnn_extractor(tensor).squeeze().numpy().reshape(1, -1)
    cnn_compressed = pca.transform(cnn_raw)[0]

    cnn_dict = {f'cnn_feat_{i}': cnn_compressed[i] for i in range(len(cnn_compressed))}
    return cnn_dict


def extract_opencv_features(image_path):
    colorfulness = calculate_hasler_susstrunk_colorfulness(image_path)
    w3c_colors_dict = extract_w3c_colors(image_path)
    symmetry, balance = calculate_horizontal_symmetry_and_balance(image_path)
    quad_leaves = get_quadtree_leaves(image_path)
    text_ratio, image_ratio = extract_layout_areas(image_path)

    cv_dict = {
        'colorfulness': colorfulness,
        'symmetry': symmetry,
        'balance': balance,
        'quadtree_leaves': quad_leaves,
        'text_area_ratio': text_ratio,
        'image_area_ratio': image_ratio,
        **w3c_colors_dict
    }
    return cv_dict

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
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file.flush()
            temp_file_path = temp_file.name

        try:
            with st.spinner("Analyzing semantics and visual complexity..."):
                feat_cnn_dict = extract_cnn_features(temp_file_path)
                feat_cv_dict = extract_opencv_features(temp_file_path)

                # Combine dictionaries
                full_features_dict = {**feat_cnn_dict, **feat_cv_dict}

                # Align columns explicitly using the scaler's training columns
                feature_names = list(scaler.feature_names_in_)
                input_df = pd.DataFrame([full_features_dict])[feature_names]

                # Scale features
                hybrid_scaled = scaler.transform(input_df)

                # Predict
                prediction = ml_model.predict(hybrid_scaled)
                pred_flat = np.ravel(prediction)

            final_pred = pred_flat[0] / 7.0 * 10.0

            st.success("Analysis complete!")
            st.subheader("Predicted Human Ratings")

            st.metric(
                label="Predicted Aesthetic Score",
                value=f"{final_pred:.2f}/10.0"
            )

        except Exception as err:
            st.error(f"Prediction failed: {err}")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)