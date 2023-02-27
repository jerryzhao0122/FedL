'''
time:2022/1/24
author:GuoZhenyuan
'''
from cv2 import transform
import torch
import torchvision 
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np

def get_dataset(params):
    if params.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root="/home/featurize/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_dataset = torchvision.datasets.MNIST(
            root="/home/featurize/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    elif params.dataset == 'fmnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root="/home/featurize/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="/home/featurize/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    elif params.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root="/home/featurize/data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="/home/featurize/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    else:
        pass
    # elif params.dataset == 'cifar10' and params.model == 'alexnet':
    #     tf = torchvision.transforms.Compose(
    #         [torchvision.transforms.Resize([227,227]),
    #         torchvision.transforms.ToTensor()]
    #     ) 
    #     train_dataset = torchvision.datasets.CIFAR10(
    #         root="/home/featurize/data",
    #         train=True,
    #         download=True,
    #         transform=tf
    #         # transform=ToTensor()
    #     )
    #     test_dataset = torchvision.datasets.CIFAR10(
    #         root="/home/featurize/data",
    #         train=False,
    #         download=True,
    #         transform=tf
    #         # transform=ToTensor()
    #     )
    return train_dataset, test_dataset

def get_dataset_wbox(params):
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整数据集尺寸为 32x32
    transforms.ToTensor()  # 转换为 Tensor 格式
])
    if params.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root="/home/featurize/data",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="/home/featurize/data",
            train=False,
            download=True,
            transform=transform
        )
    elif params.dataset == 'fmnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root="/home/featurize/data",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="/home/featurize/data",
            train=False,
            download=True,
            transform=transform
        )
    else:
        pass
    # elif params.dataset == 'cifar10' and params.model == 'alexnet':
    #     tf = torchvision.transforms.Compose(
    #         [torchvision.transforms.Resize([227,227]),
    #         torchvision.transforms.ToTensor()]
    #     ) 
    #     train_dataset = torchvision.datasets.CIFAR10(
    #         root="/home/featurize/data",
    #         train=True,
    #         download=True,
    #         transform=tf
    #         # transform=ToTensor()
    #     )
    #     test_dataset = torchvision.datasets.CIFAR10(
    #         root="/home/featurize/data",
    #         train=False,
    #         download=True,
    #         transform=tf
    #         # transform=ToTensor()
    #     )
    # train_dataset = covmnisttocifar10(train_dataset)
    # test_dataset = covmnisttocifar10(test_dataset)
    # print(train_dataset)


    return train_dataset, test_dataset


def covmnisttocifar10(dataset):

    all_data=[]
    all_target=[]
    # dd = transform(dataset.train_data)
    for i,j in dataset:
        img = i.repeat(3, 1, 1)
        all_data.append(img)
        all_target.append(j)

    new_dataset_data = torch.stack(all_data)
    new_dataset_target = torch.tensor(all_target)

    new_dataset = TensorDataset(new_dataset_data,new_dataset_target)
    return new_dataset