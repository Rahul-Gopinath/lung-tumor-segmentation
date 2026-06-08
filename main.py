import os
import numpy as np
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Spacingd, RandCropByPosNegLabeld
from monai.visualize import matshow3d

# 1. Pipeline configuration (Keeps the full training transforms)
view_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
    # Crops a clean 64x64x64 patch centered around positive (tumor) or negative tissue
    RandCropByPosNegLabeld(
        keys=["image", "label"], 
        label_key="label", 
        spatial_size=(96, 96, 96), 
        pos=3, 
        neg=1, 
        num_samples=1 # Better try higher values for this reason - By increasing num_samples,
                      # you are extracting maximum value out of a single disk-read operation, 
                      # drastically reducing data loading bottlenecks.
    )
])

# 2. Load the dataset 
lung_dataset = DecathlonDataset(
    root_dir="./data",
    task="Task06_Lung",
    section="training",
    transform=view_transforms,
    download=False,
    cache_rate=0.0  # Safe memory footprint
)

# 3. Pull a sample patch and UNPACK THE LIST
# Because RandCropByPosNegLabeld can return multiple samples, it outputs a list!
index = np.random.randint(0, len(lung_dataset))
index = 0
print(f"Selected image index: {index}")
sample_list = lung_dataset[index]
sample_patch = sample_list[0]  # Unpack the first cropped patch dictionary safely

# Extract NumPy arrays from the cropped tensors
# Shape will be exactly (96, 96, 96)
image_patch = sample_patch["image"][0].numpy()  
label_patch = sample_patch["label"][0].numpy()  

print(f"Successfully grabbed a training patch!")
print("Cropped Image Shape:", image_patch.shape)
print("Cropped Label Shape:", label_patch.shape)
print("Does this patch contain a tumor?", "Yes!" if np.any(label_patch == 1) else "No (Healthy Tissue Only)")

# --- DISPLAY THE TARGETED 96x96x96 CROP ---
# No strides needed! We show every single slice sequentially.

fig1 = plt.figure(1, figsize=(10, 10))
matshow3d(
    volume=image_patch[np.newaxis, ...],  # Shape: (1, 96, 96, 96)
    fig=fig1,
    title=f"Training Input Patch for Image {index+1} (96x96x96 Voxel Anatomy)",
    every_n=1,  # Show every slice smoothly
    frame_dim=-1,
    cmap="gray"
)

fig2 = plt.figure(2, figsize=(10, 10))
matshow3d(
    volume=label_patch[np.newaxis, ...],  # Shape: (1, 96, 96, 96)
    fig=fig2,
    title=f"Training Target Mask for Image {index+1} (96x96x96 Binary Label)",
    every_n=1,  # Show every matching slice smoothly
    frame_dim=-1,
    cmap="hot"
)

print("Opening patch validation windows...")
plt.show()