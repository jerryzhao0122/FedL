
import yaml
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset

def getBackdoorDataloader(params):
    '''
    获取后门数据
    '''
    data_dir = '/home/featurize/data'
    if params.dataset=='mnist' or params.dataset=='fmnist':
        apply_transform = transforms.Compose([
                                        transforms.Resize((28,28)),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor()
                                            ])
    elif params.dataset == 'cifar10':
        apply_transform = transforms.Compose([
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor()
        ])
    else:
        print("数据集不在已有数据集内")

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=False,
                                    transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                    transform=apply_transform)

    # 获取后门数据的索引值
    with open(f'./configs/poison_images.yaml','r') as f:
        poisson = yaml.safe_load(f) 

    backdoor_img_index = poisson['poison_images']

    backdoor_dataset = [train_dataset[i] for i in backdoor_img_index]
    # for i in backdoor_img_index:
    backdoor_dataset_ = []
    for i in range(len(backdoor_dataset)):
        tmp = (backdoor_dataset[i][0],7)
        backdoor_dataset_.append(tmp)
    
    # backdoor_img = torch.Tensor([train_dataset[i][0] for i in backdoor_img_index])
    # backdoor_label = torch.Tensor([1 for i in backdoor_img_index])
    # backdoor_dataset = TensorDataset(backdoor_img,backdoor_label)
    backdoor_dataloader = DataLoader(backdoor_dataset_,batch_size=params.client_bs)
    
    return backdoor_dataloader

def dataReplace(params,benigns,malicious):
    bengins_data, benigns_target = benigns
    malicious_data,malicious_target = [i for i in malicious][0]

    idx = np.random.choice([i for i in range(64)],size=params.backdoor_replace_data_num)

    sub_malicious_data = malicious_data[idx]
    sub_malicious_target = malicious_target[idx]

    new_data = torch.cat([sub_malicious_data,bengins_data[params.backdoor_replace_data_num:]])   
    new_target = torch.cat([sub_malicious_target,benigns_target[params.backdoor_replace_data_num:]])

    return new_data,new_target