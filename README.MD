# Attention Gate Residual U-Net for Brain Tumor Segmentation

Implementation of U-Net in combination with attention gate (AG) for brain tumor segmentation. U-Net is also available

## Environment

```
python = 3.6
pytorch = 1.3.1
torchvision = 0.4.2
```

## Training

```
!python train.py --tag 'your tag' \
                 --batch_size 16 \
                 --data_root 'root of trainset' \
                 --log_root 'root to save logs' \
                 --backbone 'ResNet50 or VGG16'\
                 --attention 0 or 1\
                 --pretrained '(Optional) pretrained model path'
```

## Notes

File architecture:
```
net/att-unet.py         Implementation of attentional U-Net
net/attention.py        Implementation of attention gate
net/unet-original.py    Implementation of original U-Net structure
net/resnet.py           Implementation of Residual Blocks
net/vgg.py              Implementation of VGG-16

utils/dataloader.py     Dataloader with on-the-fly data augmentation
utils/loss.py           Loss functions and Metrics

Train.py                Start training model
evaluation.py           Evaluate trained model
```

Our data augmentation schemes are implemented in ./utils/dataloader.py. The dataloader can perform on-the-fly data augmentation, but we do not actually use that. Rather, we pre-generate a augmented dataset.
