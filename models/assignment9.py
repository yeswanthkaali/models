import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False,dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) # output_size = 32,rf=5

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # output_size = 16,rf=7

        # TRANSITION BLOCK 1
        self.tn1=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False))
        # output_size =16 ,rf=7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        #output_size=16,rf=11

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        #output_size=16,rf=15
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1,stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #output_size=8,rf=19
        self.tn2=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False))
        #output_size=8,rf=19
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()        
        )
        #output_size=8,rf=27
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False,groups=32),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1),padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()        )
        #output_size=8,rf=35
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), padding=1, stride=2,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        #output_size=4,rf=43
        self.tn3=nn.Sequential(nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(1, 1), padding=0, bias=False))
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #output_size=4,rf=59
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4) 
        )
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False))
        
    

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.tn1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.tn2(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.tn3(x)
        x=self.convblock9(x)
        x=self.convblock10(x)
        x=self.gap(x)
        x=self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

