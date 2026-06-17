from monai.networks.nets import UNet, VNet, AttentionUnet

def get_model(config, device):

    if config["model"]["architecture"] == "UNet":
        model = UNet(spatial_dims=config["model"]["spatial_dims"],
                    in_channels=config["model"]["in_channels"],
                    out_channels=config["model"]["out_channels"],
                    channels=tuple(config["model"]["channels"]),
                    strides=tuple(config["model"]["strides"]),
                    num_res_units=config["model"]["num_res_units"]).to(device)
    
    elif config["model"]["architecture"] == "AttentionUnet":
        model = AttentionUnet(spatial_dims=config["model"]["spatial_dims"],
                            in_channels=config["model"]["in_channels"],
                            out_channels=config["model"]["out_channels"],
                            channels=tuple(config["model"]["channels"]),
                            strides=tuple(config["model"]["strides"])).to(device)

    elif config["model"]["architecture"] == "VNet":
        model = VNet(spatial_dims=config["model"]["spatial_dims"],
                    in_channels=config["model"]["in_channels"],
                    out_channels=config["model"]["out_channels"]).to(device)
    
    return model