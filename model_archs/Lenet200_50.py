import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 200, bias = True)
        self.fc2 = nn.Linear(200, 50, bias = True)
        self.fcout = nn.Linear(50, 10, bias = True)
        
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fcout.weight)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fcout.bias)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcout(x)
        return x
