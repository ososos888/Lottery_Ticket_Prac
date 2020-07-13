import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class parameters:
    def __init__(self):
        self.model_type = 'empty'
        self.lr = 'empty'
        self.epochs = 'empty'
        self.batch_size = 'empty'
        self.weight_decay = 'empty'
        self.prune_per_c = 'empty'
        self.prune_per_f = 'empty'
        self.prune_per_o = 'empty'
        self.test_iter = 'empty'
        self.prune_iter = 'empty'
        self.trainset = 'empty'
        self.valset = 'empty'
        self.testset = 'empty'
        self.train_loader = 'empty'
        self.val_loader = 'empty'
        self.test_loader = 'empty'
        
        """
        @property
        def remaining_weight(self):
            return self.__remaining_weight
        @remaining_weight.setter
        def remaining_weight(self, remaining_weight):
            self.__remaining_weight = remaining_weight
        """
    # LeNet300        
    def type(self, x):
        if x == 'Lenet_300_100':
            # parameters
            self.model_type = x
            self.lr = 0.0012
            self.epochs = 50
            self.batch_size = 60
            self.weight_decay = 1.2e-3
            self.test_iter= 5
            self.prune_per_c = 1
            self.prune_per_f = 0.2
            self.prune_per_o = 0.1
            self.prune_iter = 21
            
            # dataset
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.trainset = dsets.MNIST(root='../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            self.testset = dsets.MNIST(root='../MNIST_data/',
                                    train=False,
                                    transform = transform,
                                    download=True)
            self.valset = dsets.MNIST('../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            
            # validation set 분류
            self.validation_ratio = (1/12)
            num_train = len(self.trainset)
            indices = list(range(num_train))
            # 설정한 비율만큼 분할 시의 data 갯수
            split = int(np.floor(self.validation_ratio * num_train))
            # shuffle
            np.random.shuffle(indices)
            # data 분할
            train_idx, val_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            self.train_loader = torch.utils.data.DataLoader(dataset = self.trainset,
                                                      batch_size = self.batch_size,
                                                      sampler = train_sampler,
                                                      drop_last = True)

            self.val_loader = torch.utils.data.DataLoader(dataset = self.valset,
                                                      batch_size = self.batch_size,
                                                      sampler = val_sampler,
                                                      drop_last = True)

            self.test_loader = torch.utils.data.DataLoader(dataset = self.testset,
                                                      shuffle = False,
                                                      drop_last = True)
            
        elif x == 'Lenet_250_75':
            # parameters
            self.model_type = x
            self.lr = 0.0012
            self.epochs = 50
            self.batch_size = 60
            self.weight_decay = 1.2e-3
            self.test_iter= 5
            self.prune_per_c = 1
            self.prune_per_f = 0.2
            self.prune_per_o = 0.1
            self.prune_iter = 21
            
            # dataset
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.trainset = dsets.MNIST(root='../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            self.testset = dsets.MNIST(root='../MNIST_data/',
                                    train=False,
                                    transform = transform,
                                    download=True)
            self.valset = dsets.MNIST('../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            
            # validation set 분류
            self.validation_ratio = (1/12)
            num_train = len(self.trainset)
            indices = list(range(num_train))
            # 설정한 비율만큼 분할 시의 data 갯수
            split = int(np.floor(self.validation_ratio * num_train))
            # shuffle
            np.random.shuffle(indices)
            # data 분할
            train_idx, val_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            self.train_loader = torch.utils.data.DataLoader(dataset = self.trainset,
                                                      batch_size = self.batch_size,
                                                      sampler = train_sampler,
                                                      drop_last = True)

            self.val_loader = torch.utils.data.DataLoader(dataset = self.valset,
                                                      batch_size = self.batch_size,
                                                      sampler = val_sampler,
                                                      drop_last = True)

            self.test_loader = torch.utils.data.DataLoader(dataset = self.testset,
                                                      shuffle = False,
                                                      drop_last = True)
            
        elif x == 'Lenet_200_50':
            # parameters
            self.model_type = x
            self.lr = 0.0012
            self.epochs = 50
            self.batch_size = 60
            self.weight_decay = 1.2e-3
            self.test_iter= 5
            self.prune_per_c = 1
            self.prune_per_f = 0.2
            self.prune_per_o = 0.1
            self.prune_iter = 20
            
            # dataset
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.trainset = dsets.MNIST(root='../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            self.testset = dsets.MNIST(root='../MNIST_data/',
                                    train=False,
                                    transform = transform,
                                    download=True)
            self.valset = dsets.MNIST('../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            
            # validation set 분류
            self.validation_ratio = (1/12)
            num_train = len(self.trainset)
            indices = list(range(num_train))
            # 설정한 비율만큼 분할 시의 data 갯수
            split = int(np.floor(self.validation_ratio * num_train))
            # shuffle
            np.random.shuffle(indices)
            # data 분할
            train_idx, val_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            self.train_loader = torch.utils.data.DataLoader(dataset = self.trainset,
                                                      batch_size = self.batch_size,
                                                      sampler = train_sampler,
                                                      drop_last = True)

            self.val_loader = torch.utils.data.DataLoader(dataset = self.valset,
                                                      batch_size = self.batch_size,
                                                      sampler = val_sampler,
                                                      drop_last = True)

            self.test_loader = torch.utils.data.DataLoader(dataset = self.testset,
                                                      shuffle = False,
                                                      drop_last = True)
            
        # Conv6    
        elif x == 'Conv6':
            # parameters
            self.model_type = x
            self.lr = 0.0003
            self.epochs = 50
            self.batch_size = 60
            self.weight_decay = 3e-3
            self.test_iter = 5
            self.prune_per_c = 0.15
            self.prune_per_f = 0.2
            self.prune_per_o = 0.1
            self.prune_iter = 21
            
            # dataset
            transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.247, 0.243, 0.261))
                                           ])

            self.trainset = dsets.CIFAR10('../CIFAR10/',
                                     train=True,
                                     transform = transform,
                                     download=False)

            self.valset = dsets.CIFAR10('../CIFAR10/',
                                     train=True,
                                     transform = transform,
                                     download=False)

            self.testset = dsets.CIFAR10('../CIFAR10/',
                                     train=False,
                                     transform = transform,
                                     download=False)

            # validation set 분류
            self.validation_ratio = 0.1
            num_train = len(self.trainset)
            indices = list(range(num_train))
            # 설정한 비율만큼 분할 시의 data 갯수
            split = int(np.floor(self.validation_ratio * num_train))
            # shuffle
            np.random.shuffle(indices)
            # data 분할
            train_idx, val_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            self.train_loader = torch.utils.data.DataLoader(dataset = self.trainset,
                                                      batch_size = self.batch_size,
                                                      sampler = train_sampler,
                                                      drop_last = True)

            self.val_loader = torch.utils.data.DataLoader(dataset = self.valset,
                                                      batch_size = self.batch_size,
                                                      sampler = val_sampler,
                                                      drop_last = True)

            self.test_loader = torch.utils.data.DataLoader(dataset = self.testset,
                                                      shuffle = False,
                                                      drop_last = True)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
                       'frog', 'horse', 'ship', 'truck')

        elif x == 'TestModel':
            # parameters
            self.model_type = x
            self.lr = 0.001
            self.epochs = 2
            self.batch_size = 60
            self.weight_decay = 3e-3
            self.test_iter = 2
            self.prune_per_c = 0.2
            self.prune_per_f = 0.3
            self.prune_per_o = 0.1
            self.prune_iter = 2
            
            # dataset
            # dataset
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.trainset = dsets.MNIST(root='../MNIST_data/',
                                     train=True,
                                     transform = transform,
                                     download=True)
            self.testset = dsets.MNIST(root='../MNIST_data/',
                                    train=False,
                                    transform = transform,
                                    download=True)
            self.train_loader = torch.utils.data.DataLoader(dataset = self.trainset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     drop_last=True)
            self.test_loader = torch.utils.data.DataLoader(dataset = self.testset,
                                                     shuffle=False,
                                                     drop_last=True)            

            
# model class
class Lenet_300_100(nn.Module):
    def __init__(self):
        super(Lenet_300_100, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 300, bias = True)
        self.fc2 = nn.Linear(300, 100, bias = True)
        self.fcout = nn.Linear(100, 10, bias = True)
        
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fcout.weight)
        init.uniform_(self.fc1.bias)
        init.uniform_(self.fc2.bias)
        init.uniform_(self.fcout.bias)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcout(x)
        return x
    
class Lenet_250_75(nn.Module):
    def __init__(self):
        super(Lenet_250_75, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 250, bias = True)
        self.fc2 = nn.Linear(250, 75, bias = True)
        self.fcout = nn.Linear(75, 10, bias = True)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fcout.weight)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcout(x)
        return x
    
class Lenet_200_50(nn.Module):
    def __init__(self):
        super(Lenet_200_50, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 200, bias = True)
        self.fc2 = nn.Linear(200, 50, bias = True)
        self.fcout = nn.Linear(50, 10, bias = True)
        
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fcout.weight)
        init.normal_(self.fc1.bias)
        init.normal_(self.fc2.bias)
        init.normal_(self.fcout.bias)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcout(x)
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
        self.fcout = nn.Linear(256, 10, bias = True)
    
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
        x = F.dropout(x, training = self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training = self.training)
        x = self.fcout(x)
        return x
    
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 3, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(3, 6, kernel_size = 3, stride = 1, padding = 1)

        self.fc1 = nn.Linear(14*14*6, 50, bias = True)
        self.fcout = nn.Linear(50, 10, bias = True)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fcout(x)
        return x