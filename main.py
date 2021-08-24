## import
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
import model

## parser
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DML : CIFAR10, CIFAR100')
parser.add_argument('--EPOCHS', default=200, type=int)
parser.add_argument('--BATCH_SIZE', default=64, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--expansion', default=1, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=0.0005, type=float)
parser.add_argument('--optim', default='SGD', choices=['Adam', 'RMSprop'], type=str)
parser.add_argument('--nesterov',default=True, type=bool)
parser.add_argument('--step', default=60, type=int)
parser.add_argument('--gamma', default=0.1, type=float)

parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10','CIFAR100'], type=str)
parser.add_argument('--net1', default='Resnet_32', choices=['MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net2', default='Resnet_32', choices=['MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net3', default='None' , choices=['Resnet_32','MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net4', default='None', choices=['Resnet_32','MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--download', default=True, type=bool)
parser.add_argument('--use_weight_init', default=True, type=bool)

args = parser.parse_args()

## dataload
train_loader,test_loader, num_classes = dataloader(args)

## model
net=[args.net1, args.net2, args.net3, args.net4]
for i in range(len(net)):
    if net[-1]== 'None':
        net.pop()
num_net=len(net)
models=[]
optimizers=[]
schedulers=[]
for i in range(num_net):
    if net[i] == 'Resnet_32':
        models.append(model.ResNet(num_classes,args.use_weight_init).to(DEVICE))
    elif net[i] == 'MobileNet':
        models.append(model.MobileNet.to(DEVICE))
    elif net[i] == 'InceptionV1':
        models.append(model.InceptionV1.to(DEVICE))
    elif net[i] == 'WRN_28_10' :
        models.append(model.Wide_ResNet(num_classes,args.use_weight_init).to(DEVICE))

## optimizer
for i in range(num_net):
    if args.optim == 'SGD':
        optimizers.append(optim.SGD(models[i].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov))
    elif args.optim == 'Adam':
        optimizers.append(optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=args.decay))
    elif args.optim == 'RMSprop':
        optimizers.append(optim.RMSprop(models[i].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay))
    schedulers.append(optim.lr_scheduler.StepLR(optimizers[i], step_size=args.step, gamma=args.gamma))

## loss
criterion_CE = nn.CrossEntropyLoss()
criterion_KLD = nn.KLDivLoss(reduction='batchmean')

## train_1epoch
def train_epoch(model, train_loader, optimizers):
    for i in range(num_net):
        model[i].train()
    for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
        image, label = image.to(DEVICE), label.to(DEVICE)
        output=[]
        losses=[]
        KLD_loss=[]
        CE_loss=[]
        for i in range(num_net):
            output.append(model[i](image))
        for k in range(num_net):
            CE_loss.append(criterion_CE(output[k],label))
            KLD_loss.append(0)
            for l in range(num_net):
                if not l==k:
                    KLD_loss[k]+=criterion_KLD(F.log_softmax(output[k],dim=1),
                                               F.softmax(output[l],dim=1)).item()
            loss = CE_loss[k]+KLD_loss[k]/(num_net-1)
            losses.append(loss)
        for i in range(num_net):
            optimizers[i].zero_grad()
            losses[i].backward()
            optimizers[i].step()

##evaluate
def evaluate(model, test_loader):
    for i in range(num_net):
        model[i].eval()
    test_loss = [0]*num_net
    pred = [0]*num_net
    correct = [0]*num_net
    losses = []
    test_accuracy=[0]*num_net
    KLD_loss = []
    CE_loss = []
    with torch.no_grad():
        for image, label in test_loader:
            image,label = image.to(DEVICE),label.to(DEVICE)
            output=[]
            for i in range(num_net):
                output.append(model[i](image))
            for k in range(num_net):
                CE_loss.append(criterion_CE(output[k], label))
                KLD_loss.append(0)
                for l in range(num_net):
                    if not l == k:
                        KLD_loss[k] += criterion_KLD(F.log_softmax(output[k], dim=1),
                                                     F.softmax(output[l], dim=1)).item()
                loss = CE_loss[k] + KLD_loss[k] / (num_net - 1)
                losses.append(loss)
            for i in range(num_net):
                test_loss[i]=losses[i].item()
                pred[i] = output[i].max(1, keepdim = True)[1]
                correct[i] += pred[i].eq(label.view_as(pred[i])).sum().item()
    for i in range(num_net):
        #test_loss[i] /= len(test_loader.dataset)
        test_accuracy[i] = 100.*correct[i]/len(test_loader.dataset)
    return test_loss,test_accuracy
## train
for epoch in range(1,args.EPOCHS+1):
    for i in range(num_net):
        schedulers[i].step()
    train_epoch(models,train_loader,optimizers)
    test_loss, test_accuracy = evaluate(models,test_loader)
    print('[EPOCH : {}] net1 Loss: {:.4f}, net1 Accuracy: {:.2f}% \n\t net2 Loss: {:.4f}, net2 Accuracy: {:.2f}% '.format(epoch, test_loss[0], test_accuracy[0], test_loss[1],test_accuracy[1]))
    
