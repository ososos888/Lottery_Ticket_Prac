# %%
import argparse

# %%
# Arguement Parser

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser()

#입력받을 인자값 등록
parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=60, type=int)
parser.add_argument("--start_iter", default=0, type=int)
parser.add_argument("--end_iter", default=100, type=int)
parser.add_argument("--print_freq", default=1, type=int)
parser.add_argument("--valid_freq", default=1, type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")

# 입력받은 인자값을 args에 저장
args = parser.parse_args()


# %%
print(args)
