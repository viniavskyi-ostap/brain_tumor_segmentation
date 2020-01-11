import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, true):
        """
        Implementation of generalized dice loss for multi-class semantic segmentation
        Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, 1, H, W].

        Returns:
        dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]

        true_dummy = torch.eye(num_classes)[true.squeeze(1)]
        true_dummy = true_dummy.permute(0, 3, 1, 2)
        probas = F.softmax(logits, dim=1)

        true_dummy = true_dummy.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_dummy, dims)
        cardinality = torch.sum(probas + true_dummy, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, beta1, beta2, eps):
        super(CombinedLoss, self).__init__()

        self.beta1 = beta1
        self.beta2 = beta2

        self.dice_loss = GeneralizedDiceLoss(eps)
        self.mse_loss = nn.MSELoss()

    @staticmethod
    def kl_loss(mu, sigma):
        return -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

    def forward(self, y_pred, y_true, X_pred=None, X_true=None, mu=None, sigma=None):
        loss = self.dice_loss(y_pred, y_true)

        if X_pred is not None:
            loss += self.beta1 * self.mse_loss(X_pred, X_true)
            loss += self.beta2 * CombinedLoss.kl_loss(mu, sigma)

        return loss


def get_loss(loss):
    """ Creates loss from config
        Args:
            loss (dict): dictionary of loss configuration:
            - name (str): loss name
            and other configs for specified loss
    """
    loss_name = loss['name']
    if loss_name == 'categorical_cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'generalized_dice':
        return GeneralizedDiceLoss(float(loss['smooth']))
    elif loss_name == 'combined':
        return CombinedLoss(float(loss['beta1']), float(loss['beta2']), float(loss['smooth']))
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
