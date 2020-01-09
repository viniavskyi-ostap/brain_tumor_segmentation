import numpy as np
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, config, train_dl, val_dl, device):
        self.model = model
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device

    def train(self):
        self._init_params()
        self.model.to(self.device)
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(epoch)
            val_loss = self._validate(epoch)

            print(f"Epoch: {epoch}; train loss = {train_loss}; validation loss = {val_loss}")

    def _init_params(self):
        self.epochs = self.config["num_epochs"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _run_epoch(self, epoch):
        self.model.train()
        counter = 0
        losses = []

        for X, y in self.train_dl:
            self.model.zero_grad()

            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)

            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            if counter % 10 == 0:
                print(f"Step: {counter}; loss = {loss}")
            losses.append(loss.item())
            counter += 1

        return np.mean(losses)

    def _validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for X, y in self.val_dl:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)

        return loss.item()
