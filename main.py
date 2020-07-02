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

# custom librarys (model, parameters...) Lottery_Ticket_Prac/custom/utils.py
import custom.utils as cu

# %%
"""
# random seed for test
torch.manual_seed(55)
torch.cuda.manual_seed_all(55)
torch.backends.cudnn.enabled = False
"""

# %%
# train, test, prune, util function
def train(model, dataloader, optimizer, criterion, cp_mask):
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

# prune function
# pruning mask 생성 -> mask 복사 -> weight initialize -> prune 진행
def weight_init(model1, model2, c_rate, f_rate, o_rate):
    # layer별로 지정된 rate만큼 prune mask 생성
    for name, module in model1.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name = 'weight', amount = c_rate)
        if isinstance(module, nn.Linear):
            if 'out' in name:
                prune.l1_unstructured(module, name = 'weight', amount = o_rate)
            else:
                prune.l1_unstructured(module, name = 'weight', amount = f_rate)
            
    # mask 복사
    cp_mask = {}
    for name, mask in model1.named_buffers():
        cp_mask[name[:(len(name)-12)]] = mask
    # weight initialize
    for name, p in model1.named_parameters():
        if 'weight_orig' in name:
            for name2, p2 in model2.named_parameters():
                if name[0:len(name) - 5] in name2:
                    p.data = copy.deepcopy(p2.data)
        if 'bias_orig' in name:
            for name2, p2 in model2.named_parameters():
                if name[0:len(name) - 5] in name2:
                    p.data = copy.deepcopy(p2.data)
    # prune 진행
    for name, module in model1.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, name = 'weight')
        elif isinstance(module, nn.Linear):
            prune.remove(module, name = 'weight')            
    
    # gradient hook
    for name, module in model.named_modules():
        if 'fc' in name:
            module.weight.register_hook(lambda grad, name=name : grad.mul_(cp_mask[name]))
    
    optimizer = optim.Adam(model.parameters(), lr = param.lr, weight_decay = param.weight_decay)
    
    # copy된 mask return
    return cp_mask, optimizer

# visdom append plot
def visdom_plot(loss_plot, num, loss_value, name):
    vis.line(X = num,
            Y = loss_value,
            win = loss_plot,
            name = str(name),
            update = 'append'
            )
    
def result_plot():
    x = []
    for i in range(param.epochs+1):
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
        
# initial 정확도 확인
def zero_accu(model, dataloader, criterion, remaining_weight, vis_plt):
    accuracy, test_loss = test(model, dataloader, criterion)
    #visdom_plot(vis_plt,torch.Tensor([accuracy]), torch.Tensor([0]), str(remaining_weight))
    print('[epoch : 0] (l_loss: x.xxxxx) (t_loss: %.5f) (accu: %.4f)' % (test_loss, accuracy))
    return accuracy, test_loss

def append_result_data(running_loss, test_loss, accuracy):
    result_data[0].append(running_loss)
    result_data[1].append(test_loss)
    result_data[2].append(accuracy)
    
def wcount():
    # [전체, 남은, 비율]
    fulllist = []
    for i in range (param.prune_iter):
        weight = [0, 0, 0]
        for name, p in model.named_parameters():
            if 'conv' in name:
                if 'weight' in name:
                    a = (p != 0).sum().item() + (p == 0).sum().item()
                    weight[0] += a
                    weight[1] += int(a * (((1-param.prune_per_c) ** i)))
            elif 'fc' in name:
                if 'weight' in name:
                    if 'out' in name:
                        a = (p != 0).sum().item() + (p == 0).sum().item()
                        weight[0] += a
                        weight[1] += int(a * (((1-param.prune_per_o) ** i)))
                    else:
                        a = (p != 0).sum().item() + (p == 0).sum().item()
                        weight[0] += a
                        weight[1] += int(a * (((1-param.prune_per_f) ** i)))
        fulllist.append(round(weight[1]/weight[0] * 100, 2))
    return fulllist

def result_dict():
    result = {}
    weightper = wcount()
    for i in range(param.test_iter):
        result[(i+1)] = {}
        #for j in range(len(weightper)):
            #result[(i+1)][weightper[j]] = {}
            #for z in range(param.epochs):
                #result[(i+1)][weightper[j]][z] = {}
    return result

def average_calc():
    test_result['Average of trials'] = {}
    for weight_per in test_result[1]:
        test_result['Average of trials'][weight_per] = [[],[],[]]
        for i in range(3):
            for j in range(param.epochs+1):
                test_result['Average of trials'][weight_per][i].append(0)
        for i in range(3):
            for j in range(1, param.test_iter+1):
                for k in range(param.epochs+1):
                    test_result['Average of trials'][weight_per][i][k] += test_result[j][weight_per][i][k]
            for z in range(param.epochs+1):
                test_result['Average of trials'][weight_per][i][z] /= (param.test_iter)

# %%
# cuda setting. GPU_NUM = 사용할 GPU의 번호
GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print ('Available devices :', torch.cuda.device_count())
print ('Current cuda device : %d (%s))' % (torch.cuda.current_device(), torch.cuda.get_device_name(device)))
print("cpu와 cuda 중 다음 기기로 학습함:", device, '\n')

# %%
# set model type
print("Selcet model's number\n",
      "(1 : Lenet_300_100)\n",
      "(2 : Lenet_250_75)\n",
      "(3 : Lenet_200_50)\n",
      "(4 : Conv6)\n"
     )
while True:
    x = int(input())
    if x == 1:
        print("Selected model : Lenet_300_100")
        model_type = "Lenet_300_100"
        break
    elif x == 2:
        print("Selected model : Lenet_250_75")
        model_type = "Lenet_250_75"
        break
    elif x == 3:
        print("Selected model : Lenet_200_50")
        model_type = "Lenet_200_50"
        break
    elif x == 4:
        print("Selected model : Conv6")
        model_type = "Conv6"
        break
    elif x == 100:
        print("Selected model : TestModel")
        model_type = "TestModel"
        break
    # elif x (Adding model...)
    print("Wrong value entered!")
while True:
    fname = input("Enter a file name \n")
    x = int(input("File name is [%s]. Are you sure? [1 : Yes, 2: No]\n" % fname))
    if x == 1:
        break
        
FileName = 'test_result/' + fname + '_result.txt'
temp = sys.stdout
sys.stdout = open(FileName,'w')

# %%
param = cu.parameters()
if model_type == 'Lenet_300_100':
    model = cu.Lenet_300_100().to(device)
elif model_type == 'Lenet_250_75':
    model = cu.Lenet_250_75().to(device)
elif model_type == 'Lenet_200_50':
    model = cu.Lenet_200_50().to(device)
elif model_type == 'Conv6':
    model = cu.Conv6().to(device)
elif model_type == 'TestModel':
    model = cu.TestModel().to(device)

param.type(model_type)    
model_init = copy.deepcopy(model)
criterion = nn.CrossEntropyLoss().to(device)

# change parameter (원할 경우 class에 접근하여 직접 변경)
#param.epochs = 10
#param.test_iter = 1
#param.prune_iter = 30
# model.fc1 = nn.Linear(784, 200)

trained_weights = {}
test_result = result_dict()
# parameter check
print('\n'.join("%s: %s" % item for item in param.__dict__.items()),'\n\n')
print('Model structure\n',model)

sys.stdout.close()
sys.stdout = temp

# %%
# visdom setting
vis = visdom.Visdom()
vis.close(env="main")

Tracker_type = "Accuracy_Tracker"
title = fname + "_" + Tracker_type

# make plot
vis_plt = vis.line(X=torch.Tensor(1).zero_(), Y=torch.Tensor(1).zero_(), 
                    opts=dict(title = title,
                              legend=['100.0'],
                              showlegend=True,
                              xtickmin = 0,
                              xtickmax = 50000,
                              ytickmin = 0.94,
                              ytickmax = 0.99
                             )
                   )

# %%
print("Learning start!")
for i in range(1, (param.test_iter+1)):
    print("Test_Iter (%d/%d) Start!" % (i, param.test_iter))
    temp = sys.stdout
    sys.stdout = open(FileName,'a')

    print("=====================================================================",
          "\n\nTest_Iter (%d/%d)" % (i, param.test_iter))
    
    #model, model_init, param = set_model(model_type)
    
    #param.epochs = epochs
    #param.test_iter = test_iter
    #param.prune_iter = prune_iter

    best_accu = {}
    #result_data = [[],[],[]]
    for j in range(param.prune_iter):
        result_data = [[],[],[]]
        #cp_mask = {}
        # pruning weight, mask 복사, optimizer 재설정
        # layer별 prune rate를 입력
        
        # 이 함수 prune per 지워도 작동하나 확인해보기
        cp_mask, optimizer = weight_init(model, model_init, 
                               (1 - ((1-param.prune_per_c) ** j)),
                               (1 - ((1-param.prune_per_f) ** j)),
                               (1 - ((1-param.prune_per_o) ** j))
                              )
        
        #print(model.fcout.weight)
        
        # prune 진행 후 남은 weight 수 확인
        weight_counts = weight_counter(model)
        # 총 weight 중 남은 weight의 수 저장 (visdom plot시 사용하기 위함)
        remaining_weight = weight_counts['all.weight'][3]
        
        sys.stdout.close()
        sys.stdout = temp
        print("Learning start! [Prune_iter : (%d/%d), Remaining weight : %s %%]" % (j+1 , param.prune_iter, remaining_weight))
        temp = sys.stdout
        sys.stdout = open(FileName,'a')
        
        print("Learning start! [Prune_iter : (%d/%d), Remaining weight : %s %%]" % (j+1 , param.prune_iter, remaining_weight))
        # 시작 시간 check
        #print(model.fcout.weight[0])
        # initial accuracy 확인 및 plot
        accuracy, test_loss = zero_accu(model, param.test_loader, criterion, remaining_weight, vis_plt)
        
        append_result_data(0, test_loss.item(), accuracy)
        
        best_accu[remaining_weight] = [0, 0]
        
        start_t = timeit.default_timer()
        
        for epoch in range(param.epochs):
            # model training, return training loss    
            running_loss = train(model, param.train_loader, optimizer, criterion, cp_mask)
            # val_set이 있을 경우 val_set을 통해 loss, accu를 구한다.
            if param.valset == 'empty':
                accuracy, test_loss = test(model, param.test_loader, criterion)
            else:
                accuracy, test_loss = test(model, param.val_loader, criterion)
            
            # visdom plot (plot window, x-axis, y-axis, label name)
            #visdom_plot(vis_plt, torch.Tensor([(epoch+1) * 1000]), torch.Tensor([accuracy]),
             #           remaining_weight)
            
            append_result_data(running_loss.item(), test_loss.item(), accuracy)
            
            # Appending best accuracy in list (weight_remain, epoch, accuracy)
            if best_accu[remaining_weight][1] <= accuracy:
                best_accu[remaining_weight] = [epoch, accuracy]

            print('[epoch : %d] (l_loss: %.5f) (t_loss: %.5f) (accu: %.4f)' %
                  ((epoch+1), (running_loss), (test_loss), (accuracy)))

        stop_t = timeit.default_timer()

        #print(model.fcout.weight[0])
        
        print("Finish! (Best accu: %.4f) (Time taken(sec) : %.2f) \n\n" %
              ((best_accu[remaining_weight][1]), (stop_t - start_t)))
        test_result[i][remaining_weight] = result_data
        
    #test_result[i+1][remaining_weight] = result_data    
    # iteration별 최고 정확도 확인
    best_accuracy(best_accu)
    
    sys.stdout.close()
    sys.stdout = temp
    
    
    print("Test_Iter (%d/%d) Finish!" % (i, param.test_iter))
    # 훈련된 model의 weight 저장
    trained_weights[i] = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            a = copy.deepcopy(p)
            trained_weights[i].append(a)
    #print(trained_weights[i][2])
average_calc()
result_plot()

# Save test_result, weights dictionary
dic_FileName = "test_result/" + fname + "_result_data"
dic_FileName2 = "test_result/" + fname + "_trained_weights"
torch.save(test_result, dic_FileName)
torch.save(trained_weights, dic_FileName2)
"""
with open(dic_FileName, 'wb') as f:
    pickle.dump(test_result, f)
with open(dic_FileName2, 'wb') as f2:
    pickle.dump(trained_weights, f2)
"""

# %%
temp = sys.stdout
sys.stdout = open(FileName,'a')

print("Average test data")
for name in test_result['Average of trials']:
    print("Remaining weight %.2f %%" % name)
    print("Epoch Train_loss  Test_loss  Accuracy")
    for i in range(param.epochs+1):
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