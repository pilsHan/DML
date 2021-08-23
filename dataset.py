##import
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

##transform
def trainsform(mean,std):
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])

    return transform_train,transform_test

##dateloader
def dataloader(args):
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    dir_path = os.path.join(args.data_path,args.dataset)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    if args.dataset == 'CIFAR10':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        transform_train,transform_test=trainsform(cifar10_mean,cifar10_std)
        train_loader = DataLoader(
            datasets.CIFAR10(root=dir_path, train=True, transform=transform_train,download=args.download),
            batch_size = args.BATCH_SIZE, shuffle = True, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            datasets.CIFAR10(root=dir_path, train=False, transform=transform_test),
            batch_size = args.BATCH_SIZE, shuffle = False, num_workers=args.num_workers
        )
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        transform_train, transform_test = trainsform(cifar100_mean, cifar100_std)
        train_loader = DataLoader(
            datasets.CIFAR100(root=dir_path, train=True, transform=transform_train,download=args.download),
            batch_size = args.BATCH_SIZE, shuffle = True, num_workers=args.num_workers
        )
        test_loader = DataLoader(
            datasets.CIFAR100(root=dir_path, train=False, transform=transform_test),
            batch_size = args.BATCH_SIZE, shuffle = False, num_workers=args.num_workers
        )
        num_classes = 100

    return train_loader,test_loader, num_classes
