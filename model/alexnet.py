import torch
import torch.nn as nn
import torchvision

def get_mnist_alexnet():
    return torchvision.models.alexnet(num_classes=10)

class AlexNet_OneChannel(torch.nn.Module):
    def __init__(self, num_classes=10,track=True, init_weights=False):
        super(AlexNet_OneChannel, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 96, kernel_size=5, stride=2, padding=2),
                                          # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
                                          torch.nn.BatchNorm2d(96,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                          torch.nn.BatchNorm2d(256,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True))
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(384, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc3 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes=10,track=True, init_weights=False):
        super(AlexNet, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2),
                                          # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
                                          torch.nn.BatchNorm2d(96,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                          torch.nn.BatchNorm2d(256,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True))
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(384 * 2 * 2, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc3 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out