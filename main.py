import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import IncrementalPCA as ipc
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import tempfile

# Global Variables
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "ipca_model" not in st.session_state:
    st.session_state.ipca_model = None
if "train_images" not in st.session_state:
    st.session_state.train_images = []
if "train_image_names" not in st.session_state:
    st.session_state.train_image_names = []
if "image_matrix" not in st.session_state:
    st.session_state.image_matrix = None
if "target_shape" not in st.session_state:
    st.session_state.target_shape = None
if "original_train_images" not in st.session_state:
    st.session_state.original_train_images = []

# Helper: Convert PIL to OpenCV grayscale
def convert_to_grayscale(pil_img):
    open_cv_img = np.array(pil_img.convert('RGB'))[:, :, ::-1].copy()
    gray = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2GRAY)
    return gray

# Page 1: Train Model
def train_model_page():
    st.title("🧠 Train Face Recognition Model")

    uploaded_files = st.file_uploader("Upload training images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        images = []
        names = []
        for file in uploaded_files:
            img = Image.open(file)
            gray = convert_to_grayscale(img)
            images.append(gray)
            names.append(file.name)

        st.session_state.train_images = [cv2.resize(img, (100, 100)) for img in images]
        st.session_state.target_shape = (100, 100)
        st.session_state.original_train_images = images
        st.session_state.train_image_names = names

        # Display thumbnails
        st.subheader("Training Images Preview")
        cols = st.columns(5)
        for i, img in enumerate(images):
            with cols[i % 5]:
                st.image(img, caption=names[i], use_column_width=True)

        if st.button("Train Model"):
            resized = st.session_state.train_images
            flattened_images = [img.flatten() for img in resized]
            image_matrix = np.array(flattened_images)
            st.session_state.image_matrix = image_matrix

            ipca = ipc(n_components=min(max(10, len(resized)), 100), batch_size=10)
            ipca.fit(image_matrix)

            st.session_state.ipca_model = ipca
            st.session_state.model_trained = True

            # Mean image
            mean_image = np.mean(np.stack(resized), axis=0)
            st.subheader("Mean Image")
            st.image(mean_image, caption="Mean Image", use_column_width=True, clamp=True)

            st.success("✅ Model trained successfully!")


# Page 2: Test Image
def test_image_page():
    st.title("📸 Test Image Matcher")

    if not st.session_state.model_trained:
        st.warning("Please train the model first on the 'Train Model' page.")
        return

    test_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])
    if test_file is not None:
        img = Image.open(test_file)
        gray = convert_to_grayscale(img)
        gray_resized = cv2.resize(gray, st.session_state.target_shape)
        test_flat = gray_resized.flatten()

        # Project and find match
        test_pca = st.session_state.ipca_model.transform([test_flat])
        train_pca = st.session_state.ipca_model.transform(st.session_state.image_matrix)
        distances = euclidean_distances([test_pca], train_pca)
        closest_idx = np.argmin(distances)
        reconstructed = st.session_state.ipca_model.inverse_transform([test_pca]).reshape(st.session_state.target_shape)

        # Show results
        st.subheader("Original Test Image")
        st.image(gray_resized, use_column_width=True, caption="Uploaded Test Image")

        st.subheader("Reconstructed Image (Eigenface Approximation)")
        st.image(reconstructed, use_column_width=True, clamp=True, caption="Eigenface Reconstruction")

        st.subheader("Closest Match from Training Set")
        st.image(st.session_state.original_train_images[closest_idx], caption=f"Matched: {st.session_state.train_image_names[closest_idx]}", use_column_width=True)

        st.success(f"Closest match found: {st.session_state.train_image_names[closest_idx]}")


# Streamlit App Layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Test Image"])

if page == "Train Model":
    train_model_page()
elif page == "Test Image":
    test_image_page()
