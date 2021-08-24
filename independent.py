## module
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

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=0.0005, type=float)
parser.add_argument('--optim', default='SGD', choices=['Adam', 'RMSprop'], type=str)
parser.add_argument('--nesterov',default=True, type=bool)
parser.add_argument('--step', default=60, type=int)
parser.add_argument('--gamma', default=0.1, type=float)

parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10','CIFAR100'], type=str)
parser.add_argument('--independent', default='Resnet_32', choices=['MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net1', default='Resnet_32', choices=['MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net2', default='Resnet_32', choices=['MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net3', default='None' , choices=['Resnet_32','MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--net4', default='None', choices=['Resnet_32','MobileNet', 'InceptionV1','WRN_28_10'], type=str)
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--download', default=True, type=bool)
parser.add_argument('--use_weight_init', default=True, type=bool)

args = parser.parse_args()

##seed
def seed_everything(seed=2021):
    #random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

## dataload
train_loader,test_loader, num_classes = dataloader(args)

## model
if args.independent == 'Resnet_32':
    model=model.ResNet(num_classes,args.use_weight_init).to(DEVICE)
elif args.independent == 'MobileNet':
    model=model.MobileNet.to(DEVICE)
elif args.independent == 'InceptionV1':
    model=model.InceptionV1.to(DEVICE)
elif args.independent == 'WRN_28_10':
    model=model.Wide_ResNet(num_classes,args.use_weight_init).to(DEVICE)
    
## optimizer
if args.optim =='SGD':
    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
elif args.optim == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
elif args.optim == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

## loss
criterion_CE = nn.CrossEntropyLoss()

## train
def train(model, train_loader, optimizer):
    model.train()
    train_loss=0
    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct+=pred.eq(target.view_as(pred)).sum().item()
        loss=F.cross_entropy(output, target)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100.*correct/len(train_loader.dataset)
    return train_loss, train_accuracy

##evaluate
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss +=F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.*correct/len(test_loader.dataset)


    return test_loss, test_accuracy
##
for epoch in range(1, args.EPOCHS):
    scheduler.step()
    train_loss, train_accuracy= train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print('[{}] Train Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, train_loss, train_accuracy))
    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, test_loss, test_accuracy))
