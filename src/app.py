import os
import yaml
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from monai.apps import DecathlonDataset
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Spacingd
)

@st.cache_resource
def load_pipeline_and_model():
    """Loads the config and model weights once and caches them in memory."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize architecture matching your frozen config
    model_cfg = config["model"]
    model = UNet(
        spatial_dims=model_cfg["spatial_dims"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        num_res_units=model_cfg["num_res_units"]
    ).to(device)
    
    model_path = project_root / config["paths"]["model_dir"].lstrip("./") / "best_metric_model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Validation Transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"])
    ])
    
    # Load Validation Dataset
    data_dir = project_root / config["paths"]["data_dir"].lstrip("./")
    val_data = DecathlonDataset(root_dir=str(data_dir), task="Task06_Lung", section="validation", download=False).data
    val_dataset = CacheDataset(data=val_data, transform=val_transforms, cache_rate=0.0)
    
    return model, val_dataset, device, config

# --- Streamlit UI Configurations ---
st.set_page_config(page_title="3D Lung Tumor Segmentation", layout="wide")
st.title("3D Medical Inference Dashboard")
st.sidebar.header("Pipeline Controls")

try:
    model, val_dataset, device, config = load_pipeline_and_model()
    st.sidebar.success("Model and Dataset loaded perfectly!")
except Exception as e:
    st.error(f"Initialization failed. Make sure paths in config.yaml are correct. Error: {e}")
    st.stop()

# Select a validation patient case
case_indices = list(range(len(val_dataset)))
selected_case_idx = st.sidebar.selectbox(
    "Select Validation Case:", 
    options=case_indices, 
    format_func=lambda x: f"Patient Volume #{x+1}"
)

# Run Inference button
if st.sidebar.button("Run 3D Segmentation"):
    with st.spinner("Executing sliding window inference across 3D volume..."):
        batch_data = val_dataset[selected_case_idx]
        # Add batch dimension
        inputs = batch_data["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            roi_size = tuple(config["training"]["roi_size"])
            sw_batch_size = config["training"]["sw_batch_size"]
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
            preds = torch.argmax(outputs, dim=1, keepdim=True)
        
        # Save arrays into session state so we don't recalculate on slider moves
        st.session_state["img"] = inputs.cpu().numpy()[0, 0, :, :, :]
        st.session_state["label"] = batch_data["label"].cpu().numpy()[0, :, :, :]
        st.session_state["pred"] = preds.cpu().numpy()[0, 0, :, :, :]
        st.session_state["inference_done"] = True

# Display Interactive Slice Viewer once inference is completed
if st.session_state.get("inference_done", False):
    img_array = st.session_state["img"]
    label_array = st.session_state["label"]
    pred_array = st.session_state["pred"]
    
    # Try to find default slice containing the tumor mass center
    tumor_slices = np.where(label_array > 0)[2]
    default_slice = int(tumor_slices[len(tumor_slices) // 2]) if len(tumor_slices) > 0 else img_array.shape[2] // 2
    
    st.write("### Volumetric Slice Viewer")
    slice_idx = st.slider("Scroll through Z-Axis Depth:", 0, img_array.shape[2] - 1, default_slice)
    
    # Plotting panels side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array[:, :, slice_idx], cmap="gray")
    axes[0].set_title(f"CT Input (Slice {slice_idx})")
    axes[0].axis("off")
    
    axes[1].imshow(img_array[:, :, slice_idx], cmap="gray")
    axes[1].imshow(label_array[:, :, slice_idx], cmap="Reds", alpha=0.4)
    axes[1].set_title("Ground Truth Label")
    axes[1].axis("off")
    
    axes[2].imshow(img_array[:, :, slice_idx], cmap="gray")
    axes[2].imshow(pred_array[:, :, slice_idx], cmap="Blues", alpha=0.4)
    axes[2].set_title("Model Prediction")
    axes[2].axis("off")
    
    st.pyplot(fig)
else:
    st.info("Choose a patient case from the sidebar menu and hit 'Run 3D Segmentation' to evaluate the model.")