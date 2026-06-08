from monai.networks.nets import UNet

def get_model(config, device):

    model = UNet(spatial_dims=config["model"]["spatial_dims"],
                 in_channels=config["model"]["in_channels"],
                 out_channels=config["model"]["out_channels"],
                 channels=tuple(config["model"]["channels"]),
                 strides=tuple(config["model"]["strides"]),
                 num_res_units=config["model"]["num_res_units"]).to(device)
    
    return model