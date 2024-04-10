import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([5, 32, 32]),
            nn.Dropout(0.1)
        ) # output_size = 30,rf=3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([10, 32, 32]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # output_size = 32,rf=5
        # TRANSITION BLOCK 1
        self.tn1=nn.Sequential(nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(1, 1), padding=0, bias=False),
                            nn.MaxPool2d(2, 2))
        # output_size = 16,rf=6
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([8, 16, 16]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #output_size=16,rf=10

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([8, 16, 16]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #output_size=16,rf=14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([10, 16, 16]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #output_size=16,rf=16
        self.tn2=nn.Sequential(nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(1, 1), padding=0, bias=False),
                            nn.MaxPool2d(2, 2))
        #output_size=8,rf=18
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([8, 8, 8]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #output_size=8,rf=26
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([8, 8, 8]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #output_size=8,rf=32
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.LayerNorm([12, 8, 8]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #output_size=8,rf=40


        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8) 
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False))
        
    

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
        x = self.gap(x)
        x=self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)