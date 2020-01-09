import yaml
import torch
from model_training.dataset import BratsDataset
from model_training.unet_addnet.trainer import Trainer
from model_training.unet_addnet.model import UNetAddNet


with open("train.yaml", "r") as config_file:
    config = yaml.full_load(config_file)

train_ds = BratsDataset(config["train"]["path"])
val_ds = BratsDataset(config["val"]["path"])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=True)

if torch.cuda.is_available():
    print("Train on GPU")
    device = torch.device("cuda")
else:
    print("Train on CPU")
    device = torch.device("cpu")

model = UNetAddNet(in_channels=4, out_channels=4)

trainer = Trainer(model, config, train_dl, val_dl, device)
