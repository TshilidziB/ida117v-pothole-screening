# OpenAI Codex was used as a support tool during development of this application.
# The group reviewed and tested the final behaviour.
from pathlib import Path
import time

import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import sobel


PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_FOLDER / "q5_outputs" / "final_pothole_pipeline.joblib"
MAX_FILE_SIZE = 10 * 1024 * 1024


def extract_image_features(image):
    """Create the same HOG, colour and edge features used in the notebook."""
    rgb_image = image.convert("RGB").resize((128, 128))
    rgb_pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    gray_pixels = rgb2gray(rgb_pixels)

    # HOG follows the scikit-image feature.hog documentation:
    # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
    # These parameters match the feature extraction used in the notebook.
    hog_features = hog(
        gray_pixels,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # Colour histograms give a short summary of light and dark colour levels.
    colour_features = []
    for channel in range(3):
        histogram, _ = np.histogram(
            rgb_pixels[:, :, channel], bins=16, range=(0, 1), density=True
        )
        colour_features.extend(histogram)

    # Sobel measurements summarise the strength of texture boundaries.
    edges = sobel(gray_pixels)
    edge_features = [
        edges.mean(),
        edges.std(),
        np.quantile(edges, 0.75),
        np.quantile(edges, 0.90),
    ]

    return np.concatenate(
        [hog_features, colour_features, edge_features]
    ).astype(np.float32)


@st.cache_resource
def load_model_bundle():
    """Load the saved scaler and classifier once for the current app session."""
    return joblib.load(MODEL_FILE)


st.set_page_config(page_title="Pothole Screening", page_icon="🛣️")
st.title("South African Pothole Image Screening")
st.write(
    "Upload one road image. The model classifies it as **pothole** or "
    "**no pothole** and reports its confidence."
)

if not MODEL_FILE.exists():
    st.error("The saved model is missing. Run the project notebook first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Road image",
    type=["jpg", "jpeg", "png"],
    help="Accepted formats: JPG, JPEG and PNG. Maximum size: 10 MB.",
)
st.caption("Project limit: 10 MB per image; minimum dimensions: 32 × 32 pixels.")

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("The image is larger than 10 MB. Please choose a smaller file.")
        st.stop()

    try:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.verify()
        uploaded_file.seek(0)
        uploaded_image = Image.open(uploaded_file)
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be read as a valid road image.")
        st.stop()

    if uploaded_image.width < 32 or uploaded_image.height < 32:
        st.error("The image is too small. Use an image of at least 32 × 32 pixels.")
        st.stop()

    st.image(uploaded_image, caption="Uploaded road image", use_container_width=True)

    prediction_start = time.perf_counter()
    model_bundle = load_model_bundle()
    features = extract_image_features(uploaded_image).reshape(1, -1)
    scaled_features = model_bundle["preprocessing"].transform(features)
    pothole_probability = float(
        model_bundle["model"].predict_proba(scaled_features)[0, 1]
    )
    threshold = float(model_bundle.get("threshold", 0.5))
    predicted_label = "pothole" if pothole_probability >= threshold else "no pothole"
    confidence = (
        pothole_probability
        if predicted_label == "pothole"
        else 1.0 - pothole_probability
    )
    prediction_seconds = time.perf_counter() - prediction_start

    st.subheader(f"Prediction: {predicted_label.title()}")
    st.metric("Model confidence", f"{confidence:.1%}")
    st.progress(pothole_probability)
    st.caption(f"Estimated pothole probability: {pothole_probability:.1%}")
    st.caption(
        f"Model: {model_bundle.get('model_name', 'selected MLP')} · "
        f"preprocessing and prediction: {prediction_seconds:.3f} seconds"
    )
    st.write(
        "The prediction uses road-surface shape, colour-distribution and "
        "edge-texture measurements. Confidence is not a pothole-severity score."
    )

st.info(
    "Responsible use: this is a screening aid, not a replacement for municipal "
    "inspection. It does not estimate severity, depth, road location or repair "
    "priority. Poor lighting, unusual cameras and roads unlike the training data "
    "may reduce reliability. A person should review important decisions."
)

