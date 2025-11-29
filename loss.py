import torch.nn.functional as F
from torch import nn
import torch 

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average, reduction='mean')

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        pred_flat = pred_flat.float()
        target_flat = target_flat.float()
        loss = self.bceloss(pred_flat, target_flat)

        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss




def structure_loss(pred, mask):
    eps = 1e-6  ##### addedd by lm
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3))) #+ eps)

    #pred = torch.clamp(pred, -10, 10)    ###added by lm 
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1) #+ eps)   #### added by lm

    return (wbce + wiou).mean()


# def structure_loss(pred, mask, eps=1e-6):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

#     wsum = weit.sum(dim=(2, 3))
#     if (wsum == 0).any():
#         print("Warning: zero weight sum detected")

#     wbce = (weit * wbce).sum(dim=(2, 3)) / (wsum + eps)

#     pred = torch.clamp(pred, -10, 10)
#     pred = torch.sigmoid(pred)

#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     diff = union - inter

#     # Detect anomalies
#     if torch.isnan(inter).any() or torch.isnan(union).any():
#         print("NaN in inter/union")
#     if (diff < 0).any():
#         print("Negative union-inter values found")

#     denom = (diff + 1).clamp(min=eps)
#     wiou = 1 - (inter + 1) / denom

#     loss_val = (wbce + wiou).mean()
#     if torch.isnan(loss_val):
#         print("Loss became NaN → debug stats:",
#               "inter", inter.min().item(), inter.max().item(),
#               "union", union.min().item(), union.max().item(),
#               "diff", diff.min().item(), diff.max().item())
#     return loss_val


def iou_score(pred, mask):
    pred = torch.sigmoid(pred)
    pred = torch.nn.functional.interpolate(pred, size=(mask.shape[2], mask.shape[3]), mode='bilinear', align_corners=False)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3)) - inter
    iou = inter / (union + 1e-6)
    return iou.mean()

def dice_score(pred, mask):
    pred = torch.sigmoid(pred)
    # pred = torch.nn.functional.resize(pred, (mask.shape[0], 1, mask.shape[2], mask.shape[3]))
    pred = torch.nn.functional.interpolate(pred, size=(mask.shape[2], mask.shape[3]), mode='bilinear', align_corners=False)
    # mask = torch.nn.functional.resize(mask, (mask.shape[0], 1, mask.shape[2], mask.shape[3]))
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    union = torch.where(union == 0, inter, union)
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return dice.mean()
