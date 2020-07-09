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
from model import Lenet300_100, Lenet250_75, Lenet200_50, TESTMODEL

# custom librarys (model, parameters...) Lottery_Ticket_Prac/custom/utils.py
import custom.utils as cu

# %%
def mk_result_dict():
    #test_result[test_iter][prune_iter][epoch]["Running_loss"]["Test_loss"]["Accuracy"]
    test_result = {}
    for test_iter in range(args.test_iters):
        test_result[test_iter+1] = {}
        for prune_iter in range(args.prune_iters):
            test_result[test_iter+1][prune_iter] = {}
            for epoch in range((args.epochs)+1):
                test_result[test_iter+1][prune_iter][epoch] = {}            
    return test_result

def save_result():
    #[test_iter][prune_iter][epoch]
    test_result2[test_iter][prune_iter][epoch+1]["Running_loss"] = running_loss
    test_result2[test_iter][prune_iter][epoch+1]["Test_loss"] = test_loss
    test_result2[test_iter][prune_iter][epoch+1]["Accuracy"] = accuracy
    #return test_result

# %%
"""
# random seed for test
torch.manual_seed(55)
torch.cuda.manual_seed_all(55)
torch.backends.cudnn.enabled = False
"""

# %%
# cuda setting. GPU_NUM = 사용할 GPU의 번호
GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Available GPU devices :', torch.cuda.device_count())
print ('Current cuda device : %d (%s))' % (torch.cuda.current_device(), torch.cuda.get_device_name(device)))

# %%
if __name__=="__main__":
    # make instance
    parser = argparse.ArgumentParser()

    # Register parameter value
    parser.add_argument("--testname", default = "TESTMODEL2", type=str, help="Check your test file name")
    parser.add_argument("--epochs",default=17, type=int, help="Epoch")
    parser.add_argument("--lr",default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int, help="Batch size")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight_decay")
    parser.add_argument("--test_iters", default=2, type=int, help="Test iterations")
    parser.add_argument("--prune_iters", default=21, type=int, help="Pruning iterations")
    parser.add_argument("--prune_per_conv", default=1, type=float, help="Prune percentage of convoultion layer")
    parser.add_argument("--prune_per_linear", default=0.2, type=float, help="Prune percentage of linear layer")
    parser.add_argument("--prune_per_out", default=0.1, type=float, help="Prune percentage of out layer")
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--validation_ratio", default = (1/12), type=float, help="Validation ratio")
    parser.add_argument("--model_arch", default="Lenet300_100", type=str, help="Lenet300_100, Lenet250_75, Lenet200_50")
    parser.add_argument("--test_type", default="test_accu", type=str, help="If you want to use validation set, enter val_accu")

    # save to args
    args = parser.parse_args()

# %%
# train, test, prune, util function
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

        # validation set classification
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_ratio * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # make data lodaer
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
    #elif: ...
    
    return train_loader, val_loader, test_loader
    
# model training function    
def train(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, label) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss / len(dataloader)
    return running_loss

# model test function return accuracy n loss
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
        # 로더 -> 배치 개수 로더.dataset -> 전체 길이, 
    return (correct/total), test_loss

# find initial accuracy
def zero_accu(model, dataloader, criterion, remaining_weight, vis_plt):
    accuracy, test_loss = test(model, dataloader, criterion)
    running_loss = 0
    #visdom_plot(vis_plt,torch.Tensor([accuracy]), torch.Tensor([0]), str(remaining_weight))
    print('[epoch : 0] (l_loss: 0) (t_loss: %.5f) (accu: %.4f)' % (test_loss, accuracy))
    print(test_iter, prune_iter)
    test_result2[test_iter][prune_iter][0]["Running_loss"] = running_loss
    test_result2[test_iter][prune_iter][0]["Test_loss"] = test_loss
    test_result2[test_iter][prune_iter][0]["Accuracy"] = accuracy
    
    return accuracy, test_loss, running_loss

# prune function. weight pruning n copied mask
def weight_prune(prune_iter):
    conv_rate = (1 - ((1-args.prune_per_conv) ** prune_iter))
    fc_rate = (1 - ((1-args.prune_per_linear) ** prune_iter))
    out_rate = (1 - ((1-args.prune_per_out) ** prune_iter))
    # make prune mask
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name = 'weight', amount = conv_rate)
        if isinstance(module, nn.Linear):
            if 'out' in name:
                prune.l1_unstructured(module, name = 'weight', amount = out_rate)
            else:
                prune.l1_unstructured(module, name = 'weight', amount = fc_rate)
            
    # mask copy   
    cpd_mask = {}
    for name, mask in model.named_buffers():
        cpd_mask[name] = mask
    
    # going prune
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, name = 'weight')
        elif isinstance(module, nn.Linear):
            prune.remove(module, name = 'weight')                

    return cpd_mask

# weight initialize and apply mask function. 
def weight_init_apply():
    # weight initialize to first model
    for name_model, param_model in model.named_parameters():
        for name_init, param_init in model_init.named_parameters():
            if name_model in name_init:
                param_model.data = copy.deepcopy(param_init.data)
                break
                
    # apply prune mask
    for name_model, param_model in model.named_parameters():
        for name_mask in cpd_mask:
            if name_model in name_mask:
                param_model.data = param_model.data.mul_(cpd_mask[name_mask])
                break
                
    # gradient hook (freeze zero-weight)
    for name_model, module in model.named_modules():
        for name_mask in cpd_mask:
            if name_model != "" and name_model in name_mask:
                hook = module.weight.register_hook(lambda grad, name_mask=name_mask : grad.mul_(cpd_mask[name_mask]))
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    
    return optimizer, hook

# visdom setting
def visdom_window():
    vis = visdom.Visdom()
    vis.close(env="main")

    Tracker_type = "Accuracy_Tracker"
    title = [args.testname, Tracker_type]
    title = "_".join(title)

    # make plot
    vis_plt = vis.line(X=torch.Tensor(1).zero_(), Y=torch.Tensor(1).zero_(), 
                        opts=dict(title = title,
                                  legend=['100.0'],
                                  showlegend=True,
                                  xtickmin = 0,
                                  xtickmax = 20000,
                                  ytickmin = 0.94,
                                  ytickmax = 0.99
                                 )
                       )
    return vis, vis_plt
# visdom plot (append)
def visdom_plot(loss_plot, num, loss_value, name):
    vis.line(X = num,
            Y = loss_value,
            win = loss_plot,
            name = str(name),
            update = 'append'
            )
def result_plot():
    x = []
    for i in range(args.epochs + 1):
        x.append(i*1000)

    for name in test_result['Average of trials']:
        visdom_plot(vis_plt, torch.Tensor(x), torch.Tensor(test_result['Average of trials'][name][2]),
                            name)

# weight count function
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

# print best accuracy in each iteration
def best_accuracy(best_accu):
    print("Maximum accuracy per weight remaining")
    """
    for i in range(len(best_accu)):
        print("Remaining weight %.1f %% " % (best_accu[i][0] * 100),
             "Epoch %d" % best_accu[i][1],
             "Accu %.4f" % best_accu[i][2])
    """
    for name in best_accu:
        print("Remaining weight %s %% " % name,
             "Epoch %d" % best_accu[name][0],
             "Accu %.4f" % best_accu[name][1])
        


def append_result_data(running_loss, test_loss, accuracy):
    result_data[0].append(running_loss)
    result_data[1].append(test_loss)
    result_data[2].append(accuracy)
    
def wcount():
    # [all, remain, per]
    fulllist = []
    for i in range (args.prune_iters):
        weight = [0, 0, 0]
        for name, p in model.named_parameters():
            if 'conv' in name:
                if 'weight' in name:
                    a = (p != 0).sum().item() + (p == 0).sum().item()
                    weight[0] += a
                    weight[1] += int(a * (((1-args.prune_per_conv) ** i)))
            elif 'fc' in name:
                if 'weight' in name:
                    if 'out' in name:
                        a = (p != 0).sum().item() + (p == 0).sum().item()
                        weight[0] += a
                        weight[1] += int(a * (((1-args.prune_per_out) ** i)))
                    else:
                        a = (p != 0).sum().item() + (p == 0).sum().item()
                        weight[0] += a
                        weight[1] += int(a * (((1-args.prune_per_linear) ** i)))
        fulllist.append(round(weight[1]/weight[0] * 100, 2))
    return fulllist

def result_dict():
    result = {}
    weightper = wcount()
    for i in range(args.test_iters):
        result[(i+1)] = {}
        #for j in range(len(weightper)):
            #result[(i+1)][weightper[j]] = {}
            #for z in range(sum_epoch):
                #result[(i+1)][weightper[j]][z] = {}
    return result

def average_calc():
    test_result['Average of trials'] = {}
    for weight_per in test_result[1]:
        test_result['Average of trials'][weight_per] = [[],[],[]]
        for i in range(3):
            for j in range(args.epochs + 1):
                test_result['Average of trials'][weight_per][i].append(0)
        for i in range(3):
            for j in range(1, args.test_iters + 1):
                for k in range(args.epochs + 1):
                    test_result['Average of trials'][weight_per][i][k] += test_result[j][weight_per][i][k]
            for z in range(args.epochs + 1):
                test_result['Average of trials'][weight_per][i][z] /= (args.test_iters)

# %%
# Filename n location
FolderLocation = "test_result"
FName_result, FName_accu = args.testname.split(), args.testname.split()
FName_result.append('result.txt'), FName_accu.append('AccuData')
FName_result, FName_accu = os.path.join(FolderLocation, "_".join(FName_result)), os.path.join(FolderLocation, "_".join(FName_accu))

# %%
temp = sys.stdout
sys.stdout = open(FName_result,'w')

# %%
model = import_model()
model_init = copy.deepcopy(model)
train_loader, val_loader, test_loader = data_loader()

criterion = nn.CrossEntropyLoss().to(device)


test_result = result_dict()
# parameter check
#print('\n'.join("%s: %s" % item for item in __dict__.items()),'\n\n')
print(args)

sys.stdout.close()
sys.stdout = temp

# %%
vis, vis_plt = visdom_window()

# %%

test_result2 = mk_result_dict()


# %%
print("Learning start!")
for test_iter in range(1, args.test_iters+1):
    print("Test_Iter (%d/%d) Start!" % (test_iter, args.test_iters))
    temp = sys.stdout
    sys.stdout = open(FName_result,'a')

    print("=====================================================================",
          "\n\nTest_Iter (%d/%d)" % (test_iter, args.test_iters))
    
    best_accu = {}

    for prune_iter in range(args.prune_iters):
        result_data = [[],[],[]]
        
        if prune_iter != 0:
            cpd_mask = weight_prune(prune_iter)
            optimizer, hook = weight_init_apply()
        else:
            model = copy.deepcopy(model_init)
            optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
            #print(model.fcout.weight)
        
        # prune 진행 후 남은 weight 수 확인
        weight_counts = weight_counter(model)
        # 총 weight 중 남은 weight의 수 저장 (visdom plot시 사용하기 위함)
        remaining_weight = weight_counts['all.weight'][3]
        
        sys.stdout.close()
        sys.stdout = temp
        print("Learning start! [Prune_iter : (%d/%d), Remaining weight : %s %%]" % (prune_iter+1 , args.prune_iters, remaining_weight))
        temp = sys.stdout
        sys.stdout = open(FName_result,'a')
        
        print("Learning start! [Prune_iter : (%d/%d), Remaining weight : %s %%]" % (prune_iter+1 , args.prune_iters, remaining_weight))
        # 시작 시간 check
        #print(model.fcout.weight[0])
        # initial accuracy 확인 및 plot
        accuracy, test_loss, running_loss = zero_accu(model, test_loader, criterion, remaining_weight, vis_plt)
            
        append_result_data(0, test_loss.item(), accuracy)
        
        best_accu[remaining_weight] = [0, 0]
        
        start_t = timeit.default_timer()
        
        for epoch in range(args.epochs):
            running_loss = train(model, train_loader, optimizer, criterion)
            
            if args.test_type == 'test_accu':
                accuracy, test_loss = test(model, test_loader, criterion)
            else:
                accuracy, test_loss = test(model, val_loader, criterion)
            
            append_result_data(running_loss.item(), test_loss.item(), accuracy)
            
            # Appending best accuracy in list (weight_remain, epoch, accuracy)
            if best_accu[remaining_weight][1] <= accuracy:
                best_accu[remaining_weight] = [epoch, accuracy]

            print('[epoch : %d] (l_loss: %.5f) (t_loss: %.5f) (accu: %.4f)' %
                  ((epoch+1), (running_loss), (test_loss), (accuracy)))
            
            save_result()
            
            
        if prune_iter != 0:
            hook.remove()
        stop_t = timeit.default_timer()

        #print(model.fcout.weight[0])
        
        print("Finish! (Best accu: %.4f) (Time taken(sec) : %.2f) \n\n" %
              ((best_accu[remaining_weight][1]), (stop_t - start_t)))
        test_result[test_iter][remaining_weight] = result_data
    #test_result[i+1][remaining_weight] = result_data    
    # iteration별 최고 정확도 확인
    best_accuracy(best_accu)
    
    sys.stdout.close()
    sys.stdout = temp
    
    
    print("Test_Iter (%d/%d) Finish!" % (test_iter, args.test_iters))

average_calc()
result_plot()

# Save test_result, weights dictionary

torch.save(test_result, FName_accu)
torch.save(test_result2, FName_accu+"2")
"""
with open(dic_FileName, 'wb') as f:
    pickle.dump(test_result, f)
with open(dic_FileName2, 'wb') as f2:
    pickle.dump(trained_weights, f2)
"""

# %%
"""
average_calc()
result_plot()
"""

# %%
"""
model.fcout.bias
"""

# %%
temp = sys.stdout
sys.stdout = open(FName_result,'a')

print("Average test data")
for name in test_result['Average of trials']:
    print("Remaining weight %.2f %%" % name)
    print("Epoch Train_loss  Test_loss  Accuracy")
    for i in range(args.epochs+1):
        print('%d     %.6f    %.6f   %.4f' % (
            i,
            test_result['Average of trials'][name][0][i],
            test_result['Average of trials'][name][1][i],
            test_result['Average of trials'][name][2][i]))

sys.stdout.close()
sys.stdout = temp

# %%
"""
test_result_ = torch.load(dic_FileName)
trained_weights_ = torch.load(dic_FileName2)
print(test_result_)
print(trained_weights_)

with open(dic_fileName, 'rb') as fin:
    test_result_ = pickle.load(fin)
with open(dic_fileName2, 'rb') as fin2:
    trained_weights_ = pickle.load(fin2)

"""