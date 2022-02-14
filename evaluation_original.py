"""
    By:     hsy
    Date:   2022/1/28
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
import os
from net.unet_original import UNet
from utils.dataloader import BraTSDataset
from utils.loss import BinaryDiceLoss, MetricsTracker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./out_models/train_BCE_141.pth"
data_root = "./eval"
save_root = "./output/BCE_141/"
log_root = os.path.join(save_root, "logs")
out_image = False

if not os.path.exists(model_path):
    raise("Model not fond")

if not os.path.exists(save_root):
    os.mkdir(save_root)

if not os.path.exists(log_root):
    os.makedirs(log_root)

net = UNet().to(device)
net.load_state_dict(torch.load(model_path))
print("{} loaded".format(model_path))

dataloader = DataLoader(BraTSDataset(data_root), batch_size=1, shuffle=False)
converter = ToPILImage()

metrics = MetricsTracker("noatt_d6c4_61", "./output/noatt_d6c4_61/logs")
loss_bce = nn.BCELoss()
loss_dice = BinaryDiceLoss(smooth=0)

for i, (img, label) in enumerate(dataloader):
    img, label = img.to(device), label.to(device)
    
    
    output = net(img)
    output = (output>0.5).float()
    # TODO: Evaluate loss
    with torch.no_grad():
        iter_dice_loss = loss_dice(output, label)
        iter_bce_loss = loss_bce(output, label)
        metrics.update(output, label, iter_dice_loss.item(), iter_bce_loss.item())
    
    # plot Result
    # 3-layers of input, label, output
    if out_image:
        plt.figure(figsize=(8,2.5))
        plt.subplot(1,5,1)
        plt.imshow(img[0,0,:,:].detach().cpu().numpy(), cmap='gray')
        plt.title('layer1')  
        plt.xticks([]),plt.yticks([])  
        plt.subplot(1,5,2)
        plt.imshow(img[0,1,:,:].detach().cpu().numpy(), cmap='gray')
        plt.title('layer2')  
        plt.xticks([]),plt.yticks([])  
        plt.subplot(1,5,3)
        plt.imshow(img[0,2,:,:].detach().cpu().numpy(), cmap='gray')
        plt.title('layer3')  
        plt.xticks([]),plt.yticks([])  
        plt.subplot(1,5,4)
        plt.imshow(label[0,0,:,:].detach().cpu().numpy(), cmap='gray')
        plt.title('label')  
        plt.xticks([]),plt.yticks([])  
        plt.subplot(1,5,5)
        plt.imshow(output[0,0,:,:].detach().cpu().numpy(), cmap='gray')
        plt.title('output')  
        plt.xticks([]),plt.yticks([])  
        plt.savefig(os.path.join(save_root, "{}.jpg".format(i)))
        plt.close()
curr_acc, curr_iou, curr_dice, curr_bce = metrics.get_metrics()
print("Acc:{} | IoU:{} | Dice:{} | BCE:{}".format(curr_acc, curr_iou, curr_dice, curr_bce))