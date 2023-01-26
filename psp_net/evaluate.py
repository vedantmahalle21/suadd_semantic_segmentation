import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, iou_score_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            # print(image.shape)
            mask_pred = net(image)
            # print(mask_pred.shape, mask_true.shape)
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                iou_score += iou_score_coeff(mask_pred, mask_true)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                for i in range(mask_pred.shape[0]): 
                    mask_true_ = F.one_hot(mask_true[i:i+1,:,:], net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred_ = F.one_hot(mask_pred[i:i+1,:,:,:].argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred_[:, 1:], mask_true_[:, 1:], reduce_batch_first=False)
                    iou_score += iou_score_coeff(mask_pred_, mask_true_)

    net.train()
    return (dice_score / max(num_val_batches, 2), iou_score / max(num_val_batches, 2))
    # return iou_score / max(num_val_batches, 1)
