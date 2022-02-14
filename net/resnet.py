"""
    By:     Hsy
    Date:   2022/1/29
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """One Convolution block in a layer
        Two types:
        1. First block of several in a layer: output = output + dsamp(identity)
        2. Others: output = output + identity

        Process: Conv1x1 (in_channel->mid_channel), Conv3x3 Stride2(in_channel->mid_channel), Conv1x1(in_channel->out_channel). 
    
    Args:
        in_channel: input channel
        mid_channel: mid way channel out_channel = 4 * mid_channel
    """
    
    def __init__(self, in_channel, mid_channel, stride = 1, dSampScheme=None):
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channel, mid_channel, 3, stride, padding=1, bias=False), # Conv3x3 shrink
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channel, mid_channel * 4, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channel * 4),
            nn.ReLU(inplace=False)
        )
        self.stride = stride
        self.dSampScheme = dSampScheme
        self.out_ReLU = nn.ReLU(inplace=False)

        
    def forward(self, input):
        identity = input
        out = self.convs(input)
        
        if self.dSampScheme is not None:
            identity = self.dSampScheme(identity)
        out += identity
        
        return self.out_ReLU(out)
    
class ResNet(nn.Module):
    """ResNet

    Args:
        block_num_list: Number of Conv Blocks in each layer. [3, 4, 6, 3] for ResNet50
    """
    def __init__(self, block_num_list=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        # in: (3, 240, 240) out: (64, 240, 240)
        self.block_num_list = block_num_list
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # in: (64, 120, 120) out: (64, 60, 60) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        # in: (64, 60, 60) out: (256, 60, 60) 
        self.conv2_x = self.makeLayer(64, 64, block_num_list[0])
        # in: (256, 60, 60) out: (512, 30, 30)
        self.conv3_x = self.makeLayer(256, 128, block_num_list[0], 2)
        # in: (512, 30, 30) out: (1024, 15, 15)
        self.conv4_x = self.makeLayer(512, 256, block_num_list[0], 2)
        # in: (1024, 15, 15) out: (2048, 8, 8)
        self.conv5_x = self.makeLayer(1024, 512, block_num_list[0], 2)
        
        # NO FCN STRUCTURE
        # self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(512 * 4, class_num)
        
    
    def makeLayer(self, prev_channel, mid_channel, block_num, stride=1):
        dSampScheme = None
        outChannel = 4 * mid_channel
        blockList = []
        # stride == 1: conv2_x from max pooling(64), match prev_channel != mid_channel * 4
        # stride == 2: following layers.
        if stride != 1 or prev_channel != mid_channel * 4:
            dSampScheme = nn.Sequential(
                nn.Conv2d(prev_channel, mid_channel*4, 1, stride, bias=False),
                nn.BatchNorm2d(outChannel)
            )
        
        convBlock = ConvBlock(prev_channel, mid_channel, stride, dSampScheme)
        blockList.append(convBlock)
        for i in range(1, block_num):
            identityBlock = ConvBlock(outChannel, mid_channel)
            blockList.append(identityBlock)

        return nn.Sequential(*blockList)
    
    def forward(self, input):
        """
        torch.Size([2, 64, 240, 240])
        torch.Size([2, 256, 120, 120])
        torch.Size([2, 512, 60, 60])
        torch.Size([2, 1024, 30, 30])
        torch.Size([2, 2048, 15, 15])
        """
        f1 = self.conv1(input)
        f2 = self.conv2_x(self.maxpool(f1))
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        # out = self.avgpool(f5)
        # out = torch.flatten(out, 1)
        # out =  self.fc(out)
        return f1, f2, f3, f4, f5
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 240, 240)
    net = ResNet()
    f1, f2, f3, f4, f5 = net(x)
    print(f1.shape)
    print(f2.shape)
    print(f3.shape)
    print(f4.shape)
    print(f5.shape)
        
        
        