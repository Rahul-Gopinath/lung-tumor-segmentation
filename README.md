# 3D Lung Tumor Segmentation using MONAI & PyTorch

## Quick Start
```bash
# Clone and install dependencies
pip install -r requirements.txt

# Run the training pipeline using the configuration layout
python -m src.train

# Run the interactive visualization tool
python -m src.visualize

The Metric: Initial Validation 3D Mean Dice: 0.168

What works: Looking at the 2D cross-section plots, the model safely finds the center of the lung tumors and matches the shape well.

The Problem: The overall 3D score is low because the model misses the very top and bottom edges of the 3D tumor volume. It struggles where the tumor fades out.

Next Steps:
- Explore better alternatives to the existent dice loss metric that doesn't cater to class imbalances.
- Feed the network more tumor patches during training \(num_samples\) to fix data imbalance.
- Upgrade from standard UNet to a better model to capture 3D edges.