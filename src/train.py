from datetime import datetime
import gc
import matplotlib.pyplot as plt
import os
import yaml

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader, decollate_batch
from monai.losses import DiceCELoss, GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from src.model import *
from src.data_utils import *


def train(config, device, model_dir, writer):

    print("Train start")

    train_transforms = define_train_transform(config)
    val_transforms = define_val_transform()

    train_dataset = get_train_dataset(config, train_transforms)
    val_dataset = get_val_dataset(config, val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0)

    model = get_model(config, device)

    max_epochs = config["training"]["max_epochs"]
    #loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    loss_function = GeneralizedDiceFocalLoss(
        to_onehot_y=True,
        include_background=False,
        softmax=True,
        lambda_gdl=1.0,
        lambda_focal=1.0
    )
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=config["training"]["learning_rate"] * 0.1)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    val_interval = config["training"]["val_interval"]
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
                    
                    roi_size = tuple(config["training"]["roi_size"])
                    sw_batch_size = config["training"]["sw_batch_size"]
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


def main():

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)

    config_path = os.path.join(project_root, "config", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")

    model_dir = os.path.join(project_root, config["paths"]["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard Logging Setup
    folder_id = "%s" % (datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=os.path.join(project_root, config["paths"]["log_dir"], folder_id))

    train(config, device, model_dir, writer)
    


if __name__=="__main__":
    main()