"""
    By:     Hsy
    Date:   2022/1/29
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_num):
        super(ConvBlock, self).__init__()
        self.dSamp = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1, padding_mode= "reflect", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )
        
        self.layers = self.makeLayers(conv_num, out_channel)
    
    def forward(self, input):
        output = self.dSamp(input)
        output = self.layers(output)
        
        return output
    
    def makeLayers(self, conv_num, out_channel):
        blockList = []
        for i in range(conv_num):
            blockList.append(nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode="reflection", bias=False))
            blockList.append(nn.BatchNorm2d(out_channel))
            blockList.append(nn.LeakyReLU(inplace=False))
        return nn.Sequential(*blockList)
    
class VGG(nn.Module):
    """
    VGG-16
        layer1: (240, 240,   3)->(240, 240,  64)    2x conv
        layer2: (240, 240,  64)->(120, 120, 128)    2x conv
        layer3: (120, 120, 128)->( 60,  60, 256)    3x conv
        layer4: ( 60,  60, 256)->( 30,  30, 512)    3x conv
        layer5: ( 30,  30, 512)->( 15,  15, 1024)   3x conv

        no final FCN layer
    """
    def __init__(self, conv_list=[2, 3, 3, 3]):
        super(VGG, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, padding_mode='reflection', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=False)
        )
        self.convList = conv_list
        self.conv2 = ConvBlock(64, 128, self.convList[0])
        self.conv3 = ConvBlock(128, 256, self.convList[1])
        self.conv4 = ConvBlock(256, 512, self.convList[2])
        self.conv5 = ConvBlock(512, 1024, self.convList[3])
    
    def forward(self, input):
        """
        5 features obtained in each layer.
        
        Return:
            f1 torch.Size([2, 64, 240, 240])
            f2 torch.Size([2, 128, 120, 120])
            f3 torch.Size([2, 256, 60, 60])
            f4 torch.Size([2, 512, 30, 30])
            f5 torch.Size([2, 1024, 15, 15])
        """
        f1 = self.conv1(input)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        return f1, f2, f3, f4, f5
        
if __name__ == "__main__":
    a = torch.randn(2, 3, 240, 240)
    net = VGG()
    f1, f2, f3, f4, f5 = net(a)
    print(f1.shape)
    print(f2.shape)
    print(f3.shape)
    print(f4.shape)
    print(f5.shape)
        
        

