##add prostate segmentation+detectuion.

# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# utils/metrics.py.

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import datetime
from sklearn.metrics import jaccard_score, f1_score

# ==== Placeholder for model loading ====
def load_model(model_name):
    st.info(f" Loading model: {model_name}")
    model = torch.nn.Identity()  # Replace with real model
    return model

# ==== Dummy segmentation function ====
def generate_mask(image, model):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), min(height, width) // 4, 255, -1)
    return mask

# ==== Dummy detection function ===
def detect_regions(image, model):
    height, width = image.shape[:2]
    x1, y1 = width // 3, height // 3
    x2, y2 = 2 * width // 3, 2 * height // 3
    detected_img = image.copy()
    cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return detected_img

# ==== Compute segmentation metrics ====
def compute_segmentation_metrics(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten() > 0
    gt_flat = gt_mask.flatten() > 0

    iou = jaccard_score(gt_flat, pred_flat)
    dice = f1_score(gt_flat, pred_flat)

    return round(iou, 4), round(dice, 4)

# ==== Save predicted mask ====
def save_predicted_mask(mask, save_dir="saved_predictions"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"pred_mask_{timestamp}.png")
    Image.fromarray(mask).save(save_path)
    return save_path

# ==== Streamlit UI ====
st.set_page_config(page_title=" Prostate Segmentation Tool", layout="wide")
st.title(" Prostate Segmentation & Detection")
st.markdown("---")

# Sidebar Controls
with st.sidebar:
    st.header("⚙ Selections")
    model_choice = st.selectbox(" Select Model", ["GAN", "Yolo_V8", "Diffusion_Model"])
    task_choice = st.radio(" Select Task", ["Detection", "Segmentation", "Both"])
    image_type = st.radio(" Select Image Type", ["MRI", "CT", "TRUS"])
    uploaded_file = st.file_uploader(" Upload Medical Image", type=["nifti", "tiff", "jpeg", "png"])
    use_ground_truth = st.checkbox(" Upload Ground Truth Mask (for metrics)")
    if use_ground_truth:
        gt_file = st.file_uploader("Upload Ground Truth Mask", type=["png", "jpeg"])

# Expand the left sidebar
st.sidebar.markdown("<style> .css-1aumxhk {width: 300px;} </style>", unsafe_allow_html=True)

# Display uploaded image
image_uploaded = False  # Track image upload status
if uploaded_file:
    image_uploaded = True
    st.subheader("📸 Uploaded Image")
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Original Image", use_column_width=True)

# Centered button to trigger task
st.markdown("### Execute Task")
st.write("Click the button below to run the selected task.")

# Dynamic button color (Red if no image, Green if image uploaded)
button_color = "danger" if not image_uploaded else "success"

# Use container_width=True to stretch the button across
run_button = st.button(" Run Task", key="run_task", type="primary", use_container_width=True, disabled=not image_uploaded)

# Button Styling
if not image_uploaded:
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #ff4b4b;  /* Red */
            color: white;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #34eb61;  /* Green */
            color: white;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Task execution on button click
if run_button:
    model = load_model(model_choice)
    output_image = image_np.copy()

    col1, col2 = st.columns(2)

    # Detection task,....
    if task_choice in ["Detection", "Both"]:
        with st.spinner(" Detecting regions..."):
            output_image = detect_regions(output_image, model)
        st.success(" Detection Complete")

    # Segmentation task,...
    if task_choice in ["Segmentation", "Both"]:
        with st.spinner(" Running segmentation..."):
            pred_mask = generate_mask(image_np, model)
            output_image[pred_mask > 0] = [255, 0, 0]
        st.success(" Segmentation Complete")

        # Show prediction and original
        col1.image(image_np, caption="Original Image", use_column_width=True)
        col2.image(output_image, caption=" Overlay with Prediction", use_column_width=True)

        # Download button for prediction mask
        st.download_button("📥 Download Predicted Mask", data=Image.fromarray(pred_mask).tobytes(),
                           file_name="predicted_mask.png")

        # Save predicted mask to disk
        saved_path = save_predicted_mask(pred_mask)
        st.info(f" Predicted mask saved at: `{saved_path}`")

        # Metrics (if ground truth provided)
        if use_ground_truth and gt_file:
            gt_mask = Image.open(gt_file).convert("L")
            gt_mask_np = np.array(gt_mask.resize(pred_mask.shape[::-1]))  # Resize to match pred

            iou, dice = compute_segmentation_metrics(pred_mask, gt_mask_np)

            st.markdown("### Metrics")
            st.metric("IoU (Jaccard)", f"{iou}")
            st.metric("Dice Score", f"{dice}")
else:
    st.warning("Please upload an image to begin.")

