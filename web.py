import os
import tempfile
import joblib
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Website Rating Predictor", page_icon="🌐", layout="centered"
)

st.title("🌐 Website Visual Rating Predictor")
st.write(
    "Upload a screenshot of any website to generate its predicted visual appeal ratings."
)

# Define the local path to your saved model file here
MODEL_PATH = "model.joblib"

@st.cache_resource
def load_injected_model(path: str):
  """Loads the developer-injected model from disk once and caches it."""
  if not os.path.exists(path):
    return None
  return joblib.load(path)


model = load_injected_model(MODEL_PATH)

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
      # Save the uploaded file to a temporary file path so models that expect a file path can read it
      suffix = os.path.splitext(uploaded_file.name)[-1].lower()
      with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

      try:
        with st.spinner("Analyzing website screenshot..."):
          # Feed the image file path directly into the model
          # If your model accepts a list of paths: model.predict([temp_file_path])
          prediction = model.predict(temp_file_path)

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