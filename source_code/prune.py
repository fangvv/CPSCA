import argparse
import os
import shutil
import time
import random
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import numpy as np
from resnet_SCA import *
from resnet import *
from compute_flops import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('-v', default='A', type=str,
                    help='version of the model')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

global args, best_prec1,acc,test_loss0, test_acc0
args = parser.parse_args()
print ("args", args)

best_prec1 = 0
cudnn.benchmark = True

def loadDatadet(infile):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp=line.split(',')
        dataset.append(temp)
    return dataset

infile='/scaler_for_prune.txt'
infile=np.array(loadDatadet(infile))
print('dataset=',infile[0])
scale1= np.array(infile, dtype=np.float32)
scale2=torch.from_numpy(scale1)
scale=scale2[0]
print(len(scale))

model = resnet(depth=56)
model = model.cuda()
dict_trained = model.state_dict().copy()

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
          .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if 1 else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")

    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(),target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))
acc= test(model)
print("Accuracy of original model：",acc)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('dict_new3.txt')

newmodel = ResNet1(depth=56)
newmodel=newmodel.cuda()
print( list(newmodel.state_dict().keys() ))
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
print( list (model.state_dict().keys() ))
dict_new = newmodel.state_dict().copy()

for each in list(dict_trained):
    if 'channel_att'  in each :
        del dict_trained[each]
    if 'spatial_att'  in each :
        del dict_trained[each]
new_list = list (newmodel.state_dict().keys() )
trained_list = list (dict_trained.keys()  )
print("new_state_dict size: {}  trained state_dict size: {}".format(len(new_list),len(trained_list)) )
print("New state_dict first 10th parameters names")
print(new_list[:20])
print("trained state_dict first 10th parameters names")
print(trained_list[:20])

for i in range(332):
    dict_new[ new_list[i] ] = dict_trained[ trained_list[i] ]

newmodel.load_state_dict(dict_new)
newmodel=newmodel.cuda()
acc = test(newmodel)

skip = {
    'A': [16, 20, 38, 54],
    'B': [16, 18, 20, 34, 38, 54],
}

prune_prob = {
    'A': [0.0, 0.0, 0.1],  #0.45M
    'B': [0.6, 0.3, 0.1],  # 0.27M
    'C': [0.6, 0.6, 0.1],  # 0.17M
    'D': [0.9, 0.9, 0.7],  # 0.04M
    'E': [0.5, 0.2, 0.1],  # 0.32M
}

layer_id = 1
cfg_mask = []
index = 0
conv = 1
cfg = []

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if layer_id <= 18:
                stage = 0
        elif layer_id <= 36:
                stage = 1
        else:
                stage = 2
        if conv==1:
            conv+=1
            continue
        elif conv!=1:
            prune_prob_stage = prune_prob[args.v][stage]
            out_channels = m.weight.data.shape[0]
            size = len(m.weight.data)
            print("size",size)
            weight_copy = scale[index:(index + size)]
            index += size
            print("index", index)
            weight_copy = weight_copy.cpu().numpy()
            num_keep = int(out_channels * (1 - prune_prob_stage))
            arg_max = np.argsort(weight_copy)
            arg_max_rev = arg_max[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            cfg.append(num_keep)
            layer_id += 1
            continue
        layer_id += 1

if args.cuda:
    newmodel.cuda()

prunemodel = ResNet1(depth=56,dataset=args.dataset, cfg=cfg)

start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
for [m0, m1] in zip(newmodel.modules(), prunemodel.modules()):
    if isinstance(m0, nn.Conv2d):
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue
    elif isinstance(m0, nn.BatchNorm2d):
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': prunemodel.state_dict()}, os.path.join(args.save, 'A.pth.tar'))


for name,parameters in prunemodel.named_parameters():
    print(name,':',parameters.size())
print("++++++++++++++++++++++++++++++++++++++++++++++++++")
for name,parameters in model.named_parameters():
    print(name,':',parameters.size())
if args.cuda:
    prunemodel.cuda()
prunemodel=prunemodel.cuda()
acc = test(prunemodel)
print("",acc)

total_params_model = sum(p_old.numel() for p_old in model.parameters())
total_params_newmodel = sum(p_old.numel() for p_old in newmodel.parameters())
total_params_prunemodel = sum(p_new.numel() for p_new in prunemodel.parameters())
print("total_params_model",total_params_model)
print("total_params_newmodel",total_params_newmodel)
print("total_params_prunemodel",total_params_prunemodel)
print("-------model-------------")
print_model_param_nums(model=model)
print_model_param_flops(model=model.cuda(),input_res=32)
print("-------newmodel-------------")
print_model_param_nums(model=newmodel)
print_model_param_flops(model=newmodel, input_res=32)
print("-------prunemodel-------------")
print_model_param_nums(model=prunemodel)
print_model_param_flops(model=prunemodel, input_res=32)