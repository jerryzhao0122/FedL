import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class resblock(nn.Module):
    def __init__(self,ch_in,ch_out,stride):
        super(resblock,self).__init__()
        self.conv_1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn_1 = nn.BatchNorm2d(ch_out,track_running_stats=False)
        self.conv_2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn_2 = nn.BatchNorm2d(ch_out,track_running_stats=False)
        self.ch_in,self.ch_out,self.stride = ch_in,ch_out,stride
        self.ch_trans = nn.Sequential()
        if ch_in != ch_out:
            self.ch_trans = nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),nn.BatchNorm2d(self.ch_out,track_running_stats=False))
        #ch_trans表示通道数转变。因为要做short_cut,所以x_pro和x_ch的size应该完全一致
        
    def  forward(self,x):
        x_pro = F.relu(self.bn_1(self.conv_1(x)))
        x_pro = self.bn_2(self.conv_2(x_pro))
        
        #short_cut:
        x_ch = self.ch_trans(x)
        out = x_pro + x_ch
        return out     

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv_1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64,track_running_stats=False))
        self.block1 = resblock(64,128,2) #长宽减半 32/2=16
        self.block2 = resblock(128,256,2) #长宽再减半 16/2=8
        self.block3 = resblock(256,512,1)
        self.block4 = resblock(512,512,1)
        self.outlayer = nn.Linear(512,10) #512*8*8=32768
        
    def forward(self,x):
        x = F.relu(self.conv_1(x))
        x = self.block1(x) 
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,[1,1])
        x = x.reshape(x.size(0),-1)
        result = self.outlayer(x)
        return result       
