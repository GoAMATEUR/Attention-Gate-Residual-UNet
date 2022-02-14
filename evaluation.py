"""
    By:     hsy
    Update: 2022/2/8
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
import os
from net.unet import UNet
from utils.dataloader import BraTSDataset
from utils.loss import BinaryDiceLoss, MetricsTracker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tag = "dice7_bce3_res_noatt_82"
model_path = "out_models/{}.pth".format(tag)
data_root = "./Val/Yes"
save_root = "./output/{}/".format(tag)
log_root = os.path.join(save_root, "logs")
net_type = "ResNet50"
out_image = True

if not os.path.exists(model_path):
    raise("Model not fond")

if not os.path.exists(save_root):
    os.makedirs(save_root)

if not os.path.exists(log_root):
    os.makedirs(log_root)

net = UNet(net_type, is_train=False).to(device)
net.load_state_dict(torch.load(model_path))
print("{} loaded".format(model_path))

dataloader = DataLoader(BraTSDataset(data_root), batch_size=1, shuffle=False)
converter = ToPILImage()

metrics = MetricsTracker(tag, log_root)
loss_bce = nn.BCELoss()
loss_dice = BinaryDiceLoss(smooth=0)

for i, (img, label) in enumerate(dataloader):
    img, label = img.to(device), label.to(device)
    
    
    output = net(img)
    
    
    # TODO: Evaluate loss
    with torch.no_grad():
        iter_dice_loss = loss_dice(output, label)
        iter_bce_loss = loss_bce(output, label)
        metrics.update(output, label, iter_dice_loss.item(), iter_bce_loss.item())
    
    
    if out_image:
        # plot Result
        # 3-layers of input, label, output
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