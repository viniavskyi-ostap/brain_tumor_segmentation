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
    elif loss_name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
