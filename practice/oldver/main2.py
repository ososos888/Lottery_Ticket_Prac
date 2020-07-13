# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init as init
import torch.nn.functional as F
import visdom
import copy
import torch.nn.utils.prune as prune
from tqdm.notebook import tqdm
import numpy as np
import timeit
import sys
import os
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

# custom model
from model_archs import Lenet300_100, Lenet250_75, Lenet200_50, TESTMODEL

"""
# random seed for test
torch.manual_seed(55)
torch.cuda.manual_seed_all(55)
torch.backends.cudnn.enabled = False
"""

# Cuda setting
GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Available GPU devices :', torch.cuda.device_count())
print ('Current cuda device : %d (%s))' % (torch.cuda.current_device(), torch.cuda.get_device_name(device)))

if __name__=="__main__":
    # Make instance
    parser = argparse.ArgumentParser()

    # Register parameter value
    parser.add_argument("--testname", default = "TEST3", type=str, help="Check your test file name")
    parser.add_argument("--epochs",default=21, type=int)
    parser.add_argument("--lr",default=1.2e-3, type=float)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--test_iters", default=3, type=int)
    parser.add_argument("--prune_iters", default=21, type=int)
    parser.add_argument("--prune_per_conv", default=1, type=float, help="Prune percentage of convoultion layer")
    parser.add_argument("--prune_per_linear", default=0.2, type=float, help="Prune percentage of linear layer")
    parser.add_argument("--prune_per_out", default=0.1, type=float, help="Prune percentage of out layer")
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--validation_ratio", default = (0), type=float, help="Validation ratio")
    parser.add_argument("--model_arch", default="Lenet300_100", type=str, help="Lenet300_100 | Lenet250_75 | Lenet200_50")
    parser.add_argument("--test_type", default="test_accu", type=str, help="test_accu | val_accu")

    # Save to args
    args = parser.parse_args()

# Import model function. return model
def import_model():
    if args.model_arch == "Lenet300_100":
        model = Lenet300_100.Lenet().to(device)
    elif args.model_arch == "Lenet250_75":
        model = Lenet300_100.Lenet().to(device)
    elif args.model_arch == "Lenet200_50":
        model = Lenet300_100.Lenet().to(device)
    elif args.model_arch == "TESTMODEL":
        model = TESTMODEL.TESTMODEL().to(device)
    return model

# Data loader function. return dataloader(train, validation, test)
def data_loader():
    if args.dataset == "mnist":
        # Load mnist dataset
        transform = transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
                    ])

        trainset = dsets.MNIST(root='../MNIST_data/',
                                 train=True,
                                 transform = transform,
                                 download=True)
        testset = dsets.MNIST(root='../MNIST_data/',
                                train=False,
                                transform = transform,
                                download=True)
        valset = dsets.MNIST('../MNIST_data/',
                                 train=True,
                                 transform = transform,
                                 download=True)

        # Validation set classification
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_ratio * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # Make data lodaer
        train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                                  batch_size = args.batch_size,
                                                  sampler = train_sampler,
                                                  drop_last = True)

        val_loader = torch.utils.data.DataLoader(dataset = valset,
                                                  batch_size = args.batch_size,
                                                  sampler = val_sampler,
                                                  drop_last = True)

        test_loader = torch.utils.data.DataLoader(dataset = testset,
                                                  shuffle = False,
                                                  drop_last = True)
    # Add aditional dataloader    
    #elif: ...
    
    return train_loader, val_loader, test_loader
    
# Model training function    
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    print("epoch start")
    for batch_idx, (data, label) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss / len(dataloader)
        print(batch_idx)
        if (batch_idx + 1) % 100 == 0:
            print(batch_idx+1)
    print("epoch end", len(dataloader), len(dataloader.dataset))
    return running_loss

# Model test function return accuracy n loss
def test(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, label)

            test_loss += loss / len(dataloader)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        # loader -> # of batch loader.dataset -> # of data 
    return (correct/total), test_loss

# Find initial accuracy
def zero_accu(model, dataloader, criterion, remaining_weight):
    accuracy, test_loss = test(model, dataloader, criterion)
    running_loss = 0
    print('[epoch : 0] (l_loss: 0.00000) (t_loss: %.5f) (accu: %.4f)' % (test_loss, accuracy))
    test_result[test_iter][prune_iter][0]["Running_loss"] = running_loss
    test_result[test_iter][prune_iter][0]["Test_loss"] = test_loss
    test_result[test_iter][prune_iter][0]["Accuracy"] = accuracy
    
    return accuracy, test_loss, running_loss

# Prune function. weight pruning n copied mask
def weight_prune(prune_iter):
    conv_rate = (1 - ((1-args.prune_per_conv) ** prune_iter))
    fc_rate = (1 - ((1-args.prune_per_linear) ** prune_iter))
    out_rate = (1 - ((1-args.prune_per_out) ** prune_iter))
    # Make prune mask
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name = 'weight', amount = conv_rate)
        if isinstance(module, nn.Linear):
            if 'out' in name:
                prune.l1_unstructured(module, name = 'weight', amount = out_rate)
            else:
                prune.l1_unstructured(module, name = 'weight', amount = fc_rate)
            
    # Copy a mask   
    cpd_mask = {}
    for name, mask in model.named_buffers():
        cpd_mask[name] = mask
    
    # Apply prune function
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, name = 'weight')
        elif isinstance(module, nn.Linear):
            prune.remove(module, name = 'weight')                

    return cpd_mask

# Weight initialize and apply mask function. 
def weight_init_apply():
    # Weight initialize to first model
    for name_model, param_model in model.named_parameters():
        for name_init, param_init in model_init.named_parameters():
            if name_model in name_init:
                param_model.data = copy.deepcopy(param_init.data)
                break
                
    # Apply prune mask
    for name_model, param_model in model.named_parameters():
        for name_mask in cpd_mask:
            if name_model in name_mask:
                param_model.data = param_model.data.mul_(cpd_mask[name_mask])
                break
                
    # Gradient hook (freeze zero-weight)
    for name_model, module in model.named_modules():
        for name_mask in cpd_mask:
            if name_model != "" and name_model in name_mask:
                hook = module.weight.register_hook(lambda grad, name_mask=name_mask : grad.mul_(cpd_mask[name_mask]))
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    
    return optimizer, hook

# Weight count function
# dict type ['Layer name' : [all, non_zero, zero, ratio]]
def weight_counter(model):
    layer_weight = {'all.weight':[0, 0, 0, 0]}
    for name, p in model.named_parameters():
        if 'weight' in name:
            remain, pruned = (p != 0).sum().item(), (p == 0).sum().item()
            layer_weight[name] = [remain+pruned, remain, pruned, round((remain/(remain+pruned))*100, 2)]           
    for i in layer_weight.keys():
        for j in range(0, 3):
            layer_weight['all.weight'][j] += layer_weight[i][j]
    layer_weight['all.weight'][3] = round(layer_weight['all.weight'][1]/layer_weight['all.weight'][0]*100, 2)
    print("------------------------------------------------------------\n",
          "Layer".center(12), "Weight".center(39), "Ratio(%)".rjust(7), sep='')
    for i in layer_weight.keys():        
        print("%s" % i.ljust(13), ":",
              ("%s (%s | %s)" % (layer_weight[i][0], layer_weight[i][1], layer_weight[i][2])).center(36),
              ("%.2f" % layer_weight[i][3]).rjust(7),
              sep=''
             )
    print("------------------------------------------------------------")
    return layer_weight

# Print best accuracy in each iteration
def best_accuracy(best_accu):
    print("Maximum accuracy per weight remaining")
    for name in best_accu:
        print("Remaining weight %s %% " % name,
             "Epoch %d" % best_accu[name][0],
             "Accu %.4f" % best_accu[name][1])

# Make result dictionary
def mk_result_dict():
    test_result = {}
    for test_iter in range(args.test_iters):
        test_result[test_iter+1] = {}
        for prune_iter in range(args.prune_iters):
            test_result[test_iter+1][prune_iter] = {}
            for epoch in range((args.epochs)+1):
                test_result[test_iter+1][prune_iter][epoch] = {}            
    return test_result

# Save result to result dict
def save_result():
    test_result[test_iter][prune_iter][epoch+1]["Running_loss"] = running_loss
    test_result[test_iter][prune_iter][epoch+1]["Test_loss"] = test_loss
    test_result[test_iter][prune_iter][epoch+1]["Accuracy"] = accuracy

# Filename n location
FolderLocation = "test_result"
FName_result, FName_accu = args.testname.split(), args.testname.split()
FName_result.append('result.txt'), FName_accu.append('AccuData')
FName_result, FName_accu = os.path.join(FolderLocation, "_".join(FName_result)), os.path.join(FolderLocation, "_".join(FName_accu))

model = import_model()
model_init = copy.deepcopy(model)
train_loader, val_loader, test_loader = data_loader()
criterion = nn.CrossEntropyLoss().to(device)
test_result = mk_result_dict()

#print('\n'.join("%s: %s" % item for item in __dict__.items()),'\n\n')
print("Learning start!")

temp = sys.stdout
sys.stdout = open(FName_result,'w')

print(args)
print("Learning start!")

sys.stdout.close()
sys.stdout = temp

for test_iter in range(1, args.test_iters+1):
    print("Test_Iter (%d/%d)" % (test_iter, args.test_iters))
    
    best_accu = {}
    
    for prune_iter in range(args.prune_iters):
        
        temp = sys.stdout
        sys.stdout = open(FName_result,'a')
        
        if prune_iter != 0:
            cpd_mask = weight_prune(prune_iter)
            optimizer, hook = weight_init_apply()
        else:
            model = copy.deepcopy(model_init)
            optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        
        # Count remaining weight
        weight_counts = weight_counter(model)
        remaining_weight = weight_counts['all.weight'][3]
             
        print("Learning start! [Test_Iter : (%d/%d), Prune_iter : (%d/%d), Remaining weight : %s %%]" %
              (test_iter, args.test_iters, prune_iter+1 , args.prune_iters, remaining_weight))

        # Find initial accuracy
        accuracy, test_loss, running_loss = zero_accu(model, test_loader, criterion, remaining_weight)            
        best_accu[remaining_weight] = [0, 0]
        start_t = timeit.default_timer()
        
        for epoch in range(args.epochs):
            running_loss = train(model, train_loader, optimizer, criterion)
            
            if args.test_type == 'test_accu':
                accuracy, test_loss = test(model, test_loader, criterion)
            else:
                accuracy, test_loss = test(model, val_loader, criterion)
            
            # Appending best accuracy in list (weight_remain, epoch, accuracy)
            if best_accu[remaining_weight][1] <= accuracy:
                best_accu[remaining_weight] = [epoch, accuracy]

            print('[epoch : %d] (l_loss: %.5f) (t_loss: %.5f) (accu: %.4f)' %
                  ((epoch+1), (running_loss), (test_loss), (accuracy)))
            
            save_result()
        test_result[test_iter][remaining_weight] = test_result[test_iter].pop(prune_iter)    
            
        if prune_iter != 0:
            hook.remove()
        stop_t = timeit.default_timer()
        
        print("Finish! (Best accu: %.4f) (Time taken(sec) : %.2f) \n\n" %
              ((best_accu[remaining_weight][1]), (stop_t - start_t)))
        
        sys.stdout.close()
        sys.stdout = temp
    # Find best accuracy in each iteration
    best_accuracy(best_accu)
    print("Test_Iter (%d/%d) Finish!" % (test_iter, args.test_iters))

print("Training End")
torch.save(test_result, FName_accu)