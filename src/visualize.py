import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import torch

from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from src.model import *
from src.data_utils import *

def main():

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)

    config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(project_root, config["paths"]["model_dir"], "best_metric_model.pth")
    output_dir = os.path.join(project_root, config["paths"]["output_dir"])

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading evaluation pipeline on device: {device}")

    val_transforms = define_val_transform()
    val_dataset = get_val_dataset(config, val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=True)

    model = get_model(config, device)

    # 4. Load trained weights safely
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model weights at {model_path}. Run training first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")

    # 5. Run inference on the first few samples
    num_images_to_save = 10
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_dataloader):
            if i >= num_images_to_save:
                break
                
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            print(f"Running inference on volume {i+1}...")
            roi_size = tuple(config["training"]["roi_size"])
            sw_batch_size = config["training"]["sw_batch_size"]
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
            
            # Convert raw logits to concrete class predictions (0 or 1) via Argmax
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            # Move data back to CPU numpy arrays for plotting
            img_array = inputs.cpu().numpy()[0, 0, :, :, :]
            label_array = labels.cpu().numpy()[0, 0, :, :, :]
            pred_array = preds.cpu().numpy()[0, 0, :, :, :]

            # Find a slice along the Z-axis where the tumor is actually present
            # This avoids rendering an uninformative, completely black slice
            tumor_indices = np.where(label_array > 0)[2]
            if len(tumor_indices) > 0:
                # Pick the middle slice of the tumor volume context
                slice_idx = tumor_indices[len(tumor_indices) // 2]
            else:
                # Fallback to middle of the scan if no label is found
                slice_idx = img_array.shape[2] // 2

            print(f"Selected slice {slice_idx} for visualization.")

            # 6. Plotting the results side-by-side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Subplot 1: Original CT scan slice
            axes[0].imshow(img_array[:, :, slice_idx], cmap="gray")
            axes[0].set_title(f"CT Input (Slice {slice_idx})")
            axes[0].axis("off")
            
            # Subplot 2: Ground Truth Annotation
            axes[1].imshow(img_array[:, :, slice_idx], cmap="gray")
            axes[1].imshow(label_array[:, :, slice_idx], cmap="Reds", alpha=0.5) # Translucent overlay
            axes[1].set_title("Ground Truth Label")
            axes[1].axis("off")
            
            # Subplot 3: Network's Prediction Mask
            axes[2].imshow(img_array[:, :, slice_idx], cmap="gray")
            axes[2].imshow(pred_array[:, :, slice_idx], cmap="Blues", alpha=0.5) # Translucent overlay
            axes[2].set_title("Model Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"validation_sample_{i+1}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Saved visualization panel to {save_path}")

    print("\nVisualization generation complete. Check the './visualizations' directory!")


if __name__ == "__main__":
    main()