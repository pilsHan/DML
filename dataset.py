##
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
##normalize
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
##dateset
def dataloader(args):
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    dir_path = os.path.join(args.data_path,args.dataset)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])
        train_loader = DataLoader(
            datasets.CIFAR10(root=dir_path, train=True, transform=transform,download=args.download),
            batch_size = args.BATCH_SIZE, shuffle = True, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            datasets.CIFAR10(root=dir_path, train=False, transform=transform),
            batch_size = args.BATCH_SIZE, shuffle = True, num_workers=args.num_workers
        )
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)
        ])
        train_loader = DataLoader(
            datasets.CIFAR100(root=dir_path, train=True, transform=transform,download=args.download),
            batch_size = args.BATCH_SIZE, shuffle = True, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            datasets.CIFAR100(root=dir_path, train=False, transform=transform),
            batch_size = args.BATCH_SIZE, shuffle = True, num_workers=args.num_workers
        )
        num_classes = 100

    return train_loader,test_loader, num_classes
