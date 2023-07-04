import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceMark7Net(nn.Module):
    def __init__(self):
        super(FaceMark7Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(73728, 1024)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bnf1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bnf2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 120)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):

        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.relu(self.bnf2(self.fc2(x)))
        # bs,_ = x.shape
        # x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
        x = self.fc3(x)
        return x

def LandmarkDetector():
    return FaceMark7Net()
