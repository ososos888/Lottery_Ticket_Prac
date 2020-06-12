# %%
import sys
import numpy as np
import random
import visdom
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler

# %%
# visdom setting
vis = visdom.Visdom()
vis.close(env="main")

# make plot
loss_plt = vis.line(Y=torch.Tensor(1).zero_(),
                    opts=dict(title = 'VGG_Loss_Tracker',
                              legend=['T_loss', 'V_loss'],
                             showlegend=True
                             )
                   )

def loss_tracker(loss_plot, loss_value, num, name):
    vis.line(X = num,
            Y = loss_value,
            win = loss_plot,
            name = name,
            update = 'append'
            )

# %%
# random seed
torch.manual_seed(555)
torch.cuda.manual_seed_all(555)
np.random.seed(555)

# %%
# CUDA 설정

GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cuda')
print("설정된 학습용 기기 :",device)

# %%
# Hyperparameter
lr = 0.001
epochs = 15
batch_size = 128

# %%
# Data 전처리
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

# Data load
trainsets = dsets.CIFAR10('../CIFAR10/',
                         train=True,
                         transform = transform,
                         download=False)

valsets = dsets.CIFAR10('../CIFAR10/',
                         train=True,
                         transform = transform,
                         download=False)

testsets = dsets.CIFAR10('../CIFAR10/',
                         train=False,
                         transform = transform,
                         download=False)

# validation set 분류
validation_ratio = 0.15
num_train = len(trainsets)
indices = list(range(num_train))
# 설정한 비율만큼 분할 시의 data 갯수
split = int(np.floor(validation_ratio * num_train))
# shuffle
np.random.shuffle(indices)
# data 분할
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(dataset = trainsets,
                                          batch_size = batch_size,
                                          sampler = train_sampler,
                                          drop_last = True)

val_loader = torch.utils.data.DataLoader(dataset = valsets,
                                          batch_size = batch_size,
                                          sampler = val_sampler,
                                          drop_last = True)

test_loader = torch.utils.data.DataLoader(dataset = testsets,
                                          batch_size = 4,
                                          shuffle = False,
                                          drop_last = True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')

# %%
model = models.vgg16()
model.classifier._modules['6'] = nn.Linear(4096, 10)
model.to(device)
print(model)

# %%

a = torch.Tensor(1,3,32,32).to(device)
out = model(a)
print(out)

# %%
# model 훈련 함수
def train(model, optimizer, criterion, DataLoader, total_batch):
    model.train()    
    running_loss = 0.0
    
    for i, data in enumerate(DataLoader, 0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss / total_batch
    return running_loss 
    
# validation loss 함수
def loss_eval(model, criterion, DataLoader, total_batch):
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(DataLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
        running_loss += loss / total_batch
    return running_loss    
    
# accuracy 계산 함수
def accu_eval(DataLoader):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(DataLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct, total

# %%
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)

# %%
# Training
t_batch = len(train_loader)
v_batch = len(val_loader)
print('Learning Start!')

for epoch in range(epochs):
    # model training
    t_running_loss = train(model, optimizer, criterion, train_loader, t_batch)

    with torch.no_grad():
    # validation loss
        v_running_loss = loss_eval(model, criterion, val_loader, v_batch)

    # validation accuracy
        correct, total = accu_eval(val_loader)

    # Plot & print
    loss_tracker(loss_plt, torch.Tensor([t_running_loss]), torch.Tensor([epoch]), 'T_loss')
    loss_tracker(loss_plt, torch.Tensor([v_running_loss]), torch.Tensor([epoch]), 'V_loss')

    print('[epoch : %d] (T_loss: %.5f) ' % (epoch + 1, t_running_loss),
          '(V_loss: %5f) ' % (v_running_loss),
          '(Val Accuract : %d %%)' % (100 * correct / total)
         )
    
print('Finished Training')

# %%
with torch.no_grad():
    correct, total = accu_eval(test_loader)
print('Accuracy (testset) : %.3f %%' % (100*correct / total))

# %%
