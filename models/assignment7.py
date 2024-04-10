import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.25)
        ) # output_size = 24,rf=5


        # TRANSITION BLOCK 1
        self.tn1=nn.Sequential(nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(1, 1), padding=0, bias=False),
                            nn.MaxPool2d(2, 2))# output_size = 12,rf=9
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(5, 5), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        #output_size=8,rf=17

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(5, 5), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        #output_size=4,rf=25

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        #output=2,rf=29
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2) 
        )
        
    

    def forward(self, x):
        x = self.convblock1(x)
        x = self.tn1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)