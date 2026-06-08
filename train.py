import os
from datetime import datetime
import gc
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    NormalizeIntensityd, 
    Spacingd,
    RandRotated,
    RandFlipd,
    RandCropByPosNegLabeld, 
    AsDiscrete
)
from monai.inferers import sliding_window_inference

def main():
    print("Train start")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")

    data_dir = "./data"
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
        RandRotated(keys=["image", "label"], range_x=15, range_y=15, range_z=15, prob=0.5, mode=["bilinear", "nearest"]),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        # Crops a clean 64x64x64 patch centered around positive (tumor) or negative tissue
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label", 
            spatial_size=(128, 128, 128), 
            pos=3, 
            neg=1, 
            num_samples=4 # Better try higher values for this reason - By increasing num_samples,
                        # you are extracting maximum value out of a single disk-read operation, 
                        # drastically reducing data loading bottlenecks.
        )
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"])])
    
    # train_dataset = DecathlonDataset(
    #     root_dir=data_dir,
    #     task="Task06_Lung",
    #     section="training",
    #     transform=train_transforms,
    #     download=False,
    #     cache_rate=1.0
    # )

    # train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    train_dataset = CacheDataset(
        data=DecathlonDataset(root_dir=data_dir, task="Task06_Lung", section="training", download=False).data,
        transform=train_transforms,
        cache_rate=1.0, # Cache 100% of the training set in system RAM
        num_workers=4
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)

    # val_dataset = DecathlonDataset(
    #     root_dir=data_dir, 
    #     task="Task06_Lung", 
    #     section="validation", 
    #     transform=val_transforms, 
    #     download=False, 
    #     cache_rate=1.0
    # )
    # val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=2)

    val_dataset = CacheDataset(
        data=DecathlonDataset(root_dir=data_dir, task="Task06_Lung", section="validation", download=False).data,
        transform=val_transforms,
        cache_rate=1.0,  # Cache 100% of validation scans in RAM
        num_workers=2
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0)

    # TensorBoard Logging Setup
    folder_id = "%s" % (datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=os.path.join("./runs/lung_tumor_experiment", folder_id))

    model = UNet(spatial_dims=3,
                 in_channels=1,
                 out_channels=2,
                 channels=(32, 64, 128, 256, 512),
                 strides=(2, 2, 2, 2),
                 num_res_units=2).to(device)
    
    max_epochs = 150
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1

    print("Beginning model training pipeline...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_dataloader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= step
        print(f"Epoch [{epoch + 1}/{max_epochs}] Complete -> Average Loss: {epoch_loss:.4f}")
        writer.add_scalar("train_loss", epoch_loss, epoch)

        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            
            with torch.no_grad():
                for val_data in val_dataloader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    roi_size = (128, 128, 128)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )
                    
                    # Convert raw model logits to clean categorical output matrices
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    
                    # Compute overlap metrics
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                # Aggregate results across validation partition
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f"--- Evaluation Epoch {epoch + 1} -> Validation Mean Dice Score: {metric:.4f} ---")
                writer.add_scalar("val_mean_dice", metric, epoch)

                # Save checkpoint if score improves
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                    print(">>> Saved New Best Performing Model Weights Checkpoint! <<<")
                
                del val_outputs, val_inputs, val_labels
                torch.cuda.empty_cache()
                gc.collect()

    print(f"\nTraining Complete! Best Mean Dice: {best_metric:.4f} achieved at Epoch {best_metric_epoch}")
    writer.close()
    


if __name__=="__main__":
    main()