import torch
import torch.nn.functional as F
from functools import partial


def dice_score(y_pred, y, epsilon, device):
    with torch.no_grad():
        num_classes = y_pred.shape[1]

        true_dummy = torch.eye(num_classes).to(device)[y.squeeze(1)]
        true_dummy = true_dummy.permute(0, 3, 1, 2)

        probas = F.softmax(y_pred, dim=1)
        _, indices = probas.max(1)
        pred_dummy = torch.eye(num_classes).to(device)[indices]
        pred_dummy = pred_dummy.permute(0, 3, 1, 2)

        true_dummy = true_dummy.type(y_pred.type())
        dims = (0,) + tuple(range(2, y.ndimension()))
        intersection = torch.sum(pred_dummy * true_dummy, dims)
        cardinality = torch.sum(pred_dummy + true_dummy, dims)
        dice_loss = (2. * intersection / (cardinality + epsilon)).mean()

    return 1 - dice_loss.item()


def get_metric(metric_name, epsilon=1e-5, device="cuda"):
    if metric_name == "dice":
        return partial(dice_score, epsilon=epsilon, device=device)
    else:
        raise ValueError(f"Metric [{metric_name}] not recognized.")
