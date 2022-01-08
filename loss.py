import torch
import torch.nn.functional as F

from typing import List

def dice_loss_torch(
    gts,
    preds,
    smooth=1e-100
) -> List:
    losses = torch.zeros((preds.shape[1]))
    preds_softmax = F.softmax(preds, dim=1)

    for i in range(preds.shape[1]):
        true = gts == i
        pred = preds_softmax[:,i,:,:]

        intersection = (true * pred).sum()
        total = pred.sum() + true.sum()

        dsc = (2*intersection + smooth) / (total + smooth)

        losses[i] = 1. - dsc

    return losses

def iou_loss_torch(
    gts,
    preds,
    smooth=1e-100
):
    losses = torch.zeros((preds.shape[1]))
    preds_softmax = F.softmax(preds, dim=1)

    for i in range(preds.shape[1]):
        true = gts == i
        pred = preds_softmax[:,i,:,:]

        intersection = (true * pred).sum()
        total = pred.sum() + true.sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)

        losses[i] = 1. - iou

    return losses

if __name__ == "__main__":
    pass