import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) 
        # RF=3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #RF=6
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        #RF=6,14

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        #RF=12,17
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #  RF=24,26
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        #   RF=24,26,56,58
        self.tn1=nn.MaxPool2d(4,4)
        #   Rf=72.74,104,106

        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 10)
        
    

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x= x+self.block1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x= x+self.block2(x)
        x=self.tn1(x)
        x = x.view(-1, 512)
        x=self.fc1(x)
        x=self.fc2(x)
        return F.log_softmax(x, dim=-1)

