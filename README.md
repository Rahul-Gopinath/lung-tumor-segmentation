# lung-tumor-segmentation

The Metric: Initial Validation 3D Mean Dice: 0.168

What works: Looking at the 2D cross-section plots, the model safely finds the center of the lung tumors and matches the shape well.

The Problem: The overall 3D score is low because the model misses the very top and bottom edges of the 3D tumor volume. It struggles where the tumor fades out.

Next Steps: 
- Feed the network more tumor patches during training (num_samples) to fix data imbalance.
- Upgrade from standard UNet to a better model to capture 3D edges.