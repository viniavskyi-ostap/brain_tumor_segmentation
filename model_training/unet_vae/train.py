import yaml
import torch
from model_training.dataset import BratsDataset
from model_training.unet_vae.trainer import TrainerVAE
from model_training.unet_addnet.model import UNetAddNet
from model_training.unet_vae.vae import VAE


with open("train.yaml", "r") as config_file:
    config = yaml.full_load(config_file)

train_ds = BratsDataset(config["train"]["path"])
val_ds = BratsDataset(config["val"]["path"])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True)

if torch.cuda.is_available():
    print("Train on GPU")
    device = torch.device("cuda:0")
else:
    print("Train on CPU")
    device = torch.device("cpu")

model = UNetAddNet(in_channels=4, out_channels=4)
vae = VAE()

trainer = TrainerVAE(model, vae, config, train_dl, val_dl, device)
trainer.train()
