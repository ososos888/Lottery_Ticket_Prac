import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 300, bias = True)
        self.fc2 = nn.Linear(300, 100, bias = True)
        self.fc3 = nn.Linear(100, 10, bias = True)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)

        self.fc1 = nn.Linear(4*4*256, 256, bias = True)
        self.fc2 = nn.Linear(256, 256, bias = True)
        self.fc3 = nn.Linear(256, 10, bias = True)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x