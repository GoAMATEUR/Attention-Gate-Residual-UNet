"""
    By:     hsy
    Update: 2022/2/7
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import os

class MetricsTracker(object):
    def __init__(self, tag, log_root):
        self.iter_IoUs = []
        self.iter_Dices = []
        self.iter_Accs = []
        self.iter_bces = []
        
        self.epoch = 0
        self.tag = tag
        self.log_root = log_root
        
        self.epoch_IoUs = []
        self.epoch_Dices = []
        self.epoch_Accs = []
        self.epoch_bces = []
    
    def update(self, input: torch.Tensor, target: torch.Tensor, dice, bce):
        
        self.iter_IoUs.append(self.IoU(input, target))
        self.iter_Accs.append(self.Acc(input, target))
        self.iter_Dices.append(dice)
        self.iter_bces.append(bce)
    
    def get_metrics(self):
        iou = np.mean(self.iter_IoUs)
        dice = np.mean(self.iter_Dices)
        acc = np.mean(self.iter_Accs)
        bce = np.mean(self.iter_bces)
        self.epoch_IoUs.append(iou)
        self.epoch_Dices.append(dice)
        self.epoch_Accs.append(acc)
        self.epoch_bces.append(bce)
        return acc, iou, dice, bce
        
    def save_logs(self):
        dict = {"tag":self.tag, "epoch": self.epoch, "eval":{"iou":self.epoch_IoUs, "acc": self.epoch_Accs, "dice":self.epoch_Dices, "bce": self.epoch_bces}}
        item = json.dumps(dict)
        with open(os.path.join(self.log_root, "{}_{}_eval.json".format(self.tag, self.epoch)), "w", encoding="utf-8") as f:
            f.write(item)
        print("Eval log {}_{}_eval.json written".format(self.tag, self.epoch))
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
        self.iter_Dices = []
        self.iter_Accs  = []
        self.iter_IoUs  = []
        self.iter_bces = []
    
    def IoU(self, input: torch.Tensor, target: torch.Tensor)->float:
        N = target.size()[0]
        target = target.view(N, -1)
        input = input.view(N, -1)
        intersection = target*input
        intersection = intersection.sum(1)
        target_sum = target.sum(1)
        input_sum = input.sum(1)
        
        ious = intersection / (target_sum + input_sum - intersection)
        iou = ious.mean().detach().cpu()
        # print(iou)
        return float(iou)

    def Acc(self, input: torch.Tensor, target: torch.Tensor)->float:
        N, C, H, W = target.size()
        target = target.view(N, -1)
        input = input.view(N, -1)
    
        
        intersection = (target == input).float()
        acc = intersection.sum() / (N * H * W)
        return float(acc.detach().cpu())

class BinaryDiceLoss(nn.Module):
    """Simple implementation of Binary Dice Loss
    
    Args:
        nn ([type]): [description]
    """
    def __init__(self, smooth=1e-5, beta=1, size_mean=True) -> None:
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.size_mean = size_mean
        self.beta = beta
        
    def forward(self, input, target):
        """
        

        Args:
            input ([type]): [1, 1, 240, 240]
            target ([type]): [1, 1, 240, 240]
        """
        
        N = target.size()[0]
        smooth = self.smooth

        input_flat = input.view(N, -1)
        targets_flat = target.view(N, -1)
        intersection = input_flat * targets_flat 
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average

#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)

#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)


#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]

#         probs = (P*class_mask).sum(1).view(-1,1)

#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)

#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
#         #print('-----bacth_loss------')
#         #print(batch_loss)


#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss

def IoU(input: torch.Tensor, target: torch.Tensor)->float:
        N = target.size()[0]
        target = target.view(N, -1)
        input = input.view(N, -1)
        intersection = target*input
        intersection = intersection.sum(1)
        target_sum = target.sum(1)
        input_sum = input.sum(1)
        
        ious = intersection / (target_sum + input_sum - intersection)
        iou = ious.mean()
        return float(iou)

if __name__ == "__main__":
    # with open("./logs/dice_test/dice_test_1_eval.json", "w", encoding="utf-8") as f:
    #     f.write("test")
    # loss_dice = BinaryDiceLoss()
    target = torch.tensor([[[[1., 1.],
                            [0., 1.]]],
                           [[[1., 1.],
                            [1., 1.]]],
                           [[[1., 1.],
                            [1., 1.]]]])
    input = torch.tensor([[[[0., 1.],
                            [0., 0.]]],
                            [[[1., 0.],
                            [0., 1.]]],
                            [[[1., 1.],
                            [1., 1.]]]])
    
    print(IoU(input, target))
    
    # N, C, H, W = target.size()
    # target = target.view(N, -1)
    # input = input.view(N, -1)
   
    
    # intersection = (target == input).float()
    # print(float(intersection.sum() / (N * H * W)))
    # # intersection = intersection.sum(1)
    # # target_sum = target.sum(1)
    # # input_sum = input.sum(1)
    
    # # ious = intersection / (target_sum + input_sum - intersection)
    # # iou = float(ious.mean())
    # # print(iou)
    
