import torch
import torch.nn.functional as F
import numpy as np

def dice_torch(
    gts,
    preds,
    smooth=1e-100
):
    dscs = torch.zeros((preds.shape[1]))
    preds_argmax = torch.argmax(preds, dim=1)

    for i in range(preds.shape[1]):
        true = gts == i
        pred = preds_argmax == i

        intersection = (true * pred).sum()
        total = pred.sum() + true.sum()

        dsc = (2*intersection + smooth) / (total + smooth)

        dscs[i] = dsc

    return dscs

def dice_np(
    gts, 
    preds,
    n_classes,
    smooth=1e-100
):
    dscs = np.zeros((n_classes))

    for i in range(n_classes):
        gt = gts == i
        pred = preds == i

        intersection = (gt * pred).sum()
        total = pred.sum() + gt.sum()

        dsc = (2*intersection + smooth) / (total + smooth)

        dscs[i] = dsc

    return dscs

def iou_torch(
    gts,
    preds,
    smooth=1e-100
):
    ious = torch.zeros((preds.shape[1]))
    preds_argmax = torch.argmax(preds, dim=1)

    for i in range(preds.shape[1]):
        true = gts == i
        pred = preds_argmax == i

        intersection = (true * pred).sum()
        total = pred.sum() + true.sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)

        ious[i] = iou

    return ious

if __name__ == "__main__":
    pass