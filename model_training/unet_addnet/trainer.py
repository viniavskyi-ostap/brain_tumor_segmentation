import numpy as np
import torch
import torch.optim as optim
import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import os

from model_training.common.losses import get_loss
from model_training.common.metrics import get_metric


class Trainer:
    def __init__(self, model, config, train_dl, val_dl, device):
        self.model = model
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.log_path = config["log_path"]

        if not os.path.exists(config["log_path"]):
            os.mkdir(config["log_path"])

    def train(self):
        self._init_params()
        self.model.to(self.device)
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(epoch)
            val_loss, metrics = self._validate()
            self.scheduler.step(val_loss)
            self._set_checkpoint(val_loss)

            print(f"\nEpoch: {epoch}; train loss = {train_loss}; validation loss = {val_loss}")

            self._write_to_tensorboard(epoch, train_loss, val_loss, metrics)

    def _save_checkpoint(self, file_prefix):
        torch.save(
            {
                'model': self.model.state_dict()
            },
            os.path.join(self.log_path, '{}.h5'.format(file_prefix)))

    def _set_checkpoint(self, val_loss):
        """ Saves model weights in the last checkpoint.
        Also, model is saved as the best model if model has the best loss
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_checkpoint(file_prefix='best')

        self._save_checkpoint(file_prefix='last')

    def _init_params(self):
        self.epochs = self.config["num_epochs"]
        self.criterion = get_loss(self.config['loss'])
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.metrics = {metric_name: get_metric(metric_name, device=self.device) for metric_name in
                        self.config["metrics"]}
        self.writer = SummaryWriter(self.config["log_path"])
        self.best_loss = float("inf")

    def _run_epoch(self, epoch):
        self.model.train()
        losses = []
        lr = self.optimizer.param_groups[0]['lr']

        status_bar = tqdm.tqdm(total=len(self.train_dl))
        status_bar.set_description(f'Epoch {epoch}, lr {lr}')

        for X, y in self.train_dl:
            self.model.zero_grad()

            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)

            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            status_bar.update()
            status_bar.set_postfix(loss=losses[-1])

        status_bar.close()
        return np.mean(losses)

    def _validate(self):
        self.model.eval()
        losses, metrics = [], defaultdict(list)

        status_bar = tqdm.tqdm(total=len(self.val_dl))

        with torch.no_grad():
            for X, y in self.val_dl:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                losses.append(loss.item())

                for metric_name in self.metrics:
                    metrics[metric_name].append(self.metrics[metric_name](y_pred, y))

                status_bar.update()
                status_bar.set_postfix(loss=losses[-1])

        status_bar.close()

        return np.mean(losses), dict(zip(metrics.keys(), map(np.mean, metrics.values())))

    def _get_scheduler(self):
        """ Creates scheduler for a given optimizer from Trainer config

            Returns:
                torch.optim.lr_scheduler._LRScheduler: optimizer scheduler
        """
        scheduler_config = self.config['scheduler']
        if scheduler_config['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                             mode=scheduler_config['mode'],
                                                             patience=scheduler_config['patience'],
                                                             factor=scheduler_config['factor'],
                                                             min_lr=scheduler_config['min_lr'])
        else:
            raise ValueError(f"Scheduler [{scheduler_config['name']}] not recognized.")
        return scheduler

    def _get_optimizer(self):
        """ Creates model optimizer from Trainer config

            Returns:
                torch.optim.optimizer.Optimizer: model optimizer
        """
        optimizer_config = self.config['optimizer']
        params = self.model.parameters()

        if optimizer_config['name'] == 'adam':
            optimizer = optim.Adam(params, lr=optimizer_config['lr'])
        elif optimizer_config['name'] == 'sgd':
            optimizer = optim.SGD(params,
                                  lr=optimizer_config['lr'],
                                  momentum=optimizer_config.get('momentum', 0),
                                  weight_decay=optimizer_config.get('weight_decay', 0))
        else:
            raise ValueError(f"Optimizer [{optimizer_config['name']}] not recognized.")
        return optimizer

    def _write_to_tensorboard(self, epoch, train_loss, val_loss, val_metrics):
        for scalar_prefix, loss in zip(('Validation', 'Train'), (train_loss, val_loss)):
            self.writer.add_scalar(f'{scalar_prefix}_Loss', loss, epoch)

        for metric_name in val_metrics:
            self.writer.add_scalar(f'Validation_{metric_name}', val_metrics[metric_name], epoch)
