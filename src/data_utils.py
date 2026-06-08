from monai.apps import DecathlonDataset
from monai.data import CacheDataset, DataLoader, decollate_batch

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

def define_train_transform(config):

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
        RandRotated(keys=["image", "label"], range_x=config["transforms"]["rotation_range_degrees"], range_y=config["transforms"]["rotation_range_degrees"], range_z=config["transforms"]["rotation_range_degrees"], prob=config["transforms"]["rotation_prob"], mode=["bilinear", "nearest"]),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=config["transforms"]["flip_prob"]),
        # Crops a clean 64x64x64 patch centered around positive (tumor) or negative tissue
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label", 
            spatial_size=tuple(config["transforms"]["spatial_size"]), 
            pos=config["transforms"]["pos_sample_ratio"], 
            neg=config["transforms"]["neg_sample_ratio"], 
            num_samples=config["transforms"]["num_samples_per_volume"] # Better try higher values for this reason - By increasing num_samples,
                        # you are extracting maximum value out of a single disk-read operation, 
                        # drastically reducing data loading bottlenecks.
        )
    ])

    return train_transforms


def define_val_transform():

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"])])
    
    return val_transforms


def get_train_dataset(config, train_transforms):

    train_dataset = CacheDataset(
        data=DecathlonDataset(root_dir=config["paths"]["data_dir"], task=config["data"]["task"], section="training", download=False).data,
        transform=train_transforms,
        cache_rate=config["data"]["cache_rate"], # Cache 100% of the training set in system RAM
        num_workers=config["data"]["num_workers_train"]
    )

    return train_dataset


def get_val_dataset(config, val_transforms):

    val_dataset = CacheDataset(
        data=DecathlonDataset(root_dir=config["paths"]["data_dir"], task=config["data"]["task"], section="validation", download=False).data,
        transform=val_transforms,
        cache_rate=config["data"]["cache_rate"],  # Cache 100% of validation scans in RAM
        num_workers=config["data"]["num_workers_val"]
    )

    return val_dataset

