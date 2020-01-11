import tqdm
import numpy as np
from model_training.unet_addnet.trainer import Trainer


class TrainerVAE(Trainer):
    def __init__(self, model, vae, config, train_dl, val_dl, device):
        super(TrainerVAE, self).__init__(model, config, train_dl, val_dl, device)
        self.vae = vae

    def _init_params(self):
        super(TrainerVAE, self)._init_params()
        self.vae.to(self.device)

    def _run_epoch(self, epoch):
        self.model.train()
        self.vae.train()
        losses = []
        lr = self.optimizer.param_groups[0]['lr']

        status_bar = tqdm.tqdm(total=len(self.train_dl))
        status_bar.set_description(f'Epoch {epoch}, lr {lr}')

        for X, y in self.train_dl:
            self.model.zero_grad()
            self.vae.zero_grad()

            X, y = X.to(self.device), y.to(self.device)
            y_pred, embedding = self.model(X)
            X_pred, mu, sigma = self.vae(embedding)

            loss = self.criterion(y_pred, y, X_pred, X, mu, sigma)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            status_bar.update()
            status_bar.set_postfix(loss=losses[-1])

        status_bar.close()
        return np.mean(losses)

    def _get_params(self):
        return list(self.model.parameters()) + list(self.vae.parameters())
