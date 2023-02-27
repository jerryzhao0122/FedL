'''
time: 2022/1/24
author: Guo Zhenyuan 
'''

import torch
from collections import defaultdict
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset, dataloader
from tqdm import tqdm

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def whitebox(dataset:datasets,params):
    num_clients = params.num_client
    batch_size = params.client_bs

    num_items = int(len(dataset) / num_clients)
    all_idxs = list(range(len(dataset)))

    all_loaders = []

    for i in range(num_clients):
        samp_idxs = range(i*num_items,(i+1)*num_items) #顺序采样，便于知道谁拥有那些数据
        # ds = [dataset[i] for i in samp_idxs] 
        dataloader = DataLoader(DatasetSplit(dataset,samp_idxs),batch_size=batch_size,shuffle=True)
        all_loaders.append(dataloader)
        all_idxs = list(set(all_idxs)-set(samp_idxs))
    
    return all_loaders

def whitebox_iid(dataset:datasets,params):
    client_num = params.num_client
    batch_size = params.client_bs

    items_num = int(len(dataset)/client_num)
    all_idxs = list(range(len(dataset)))

    all_loaders = {}
    client_samp_idxs = {}

    for i in tqdm(range(client_num)):
        samp_idxs = np.random.choice(all_idxs,items_num,replace=False)
        dataloader = DataLoader(DatasetSplit(dataset,samp_idxs),batch_size=batch_size,shuffle=True,num_workers=6)
        all_idxs = list(set(all_idxs)-set(samp_idxs))

        all_loaders[i] = dataloader
        client_samp_idxs[i] = samp_idxs
    
    return all_loaders, client_samp_idxs

def whitebox_noniid(dataset:datasets,params):
    '''
    使用dirichlet distribution来进行采样，模仿非独立同分布
    参考 How To Backdoor Federated Learning
        【AAAI】BEAS: Blockchain Enabled Asynchronous & Secure Federated Machine Learning
    time:2022/2/11
    by:zhenyuan guo
    ''' 
    all_loader = []
    dataset_class={}
    client_samp_idxs={}

    for idx, (data,target) in enumerate(dataset):
        if target in dataset_class.keys():
            dataset_class[target].append(idx)
        else:
            dataset_class[target]=[idx]
    
    class_size = [len(i) for i in dataset_class.values()]
    class_nomb = len(dataset_class.keys())
    client_idxlist = defaultdict(list)

    for i in range(class_nomb):
        random.shuffle(dataset_class[i])
        sample_numb = class_size[i] * np.random.dirichlet(np.array(params.num_client * [params.noniid_alpha]) )
        for client in range(params.num_client):
            data_nomb = int(round(sample_numb[client]))
            sampled_list = dataset_class[i][:min(class_size[i],data_nomb)]
            client_idxlist[client].extend(sampled_list)
            dataset_class[i] = dataset_class[i][min(class_size[i],data_nomb):]
    
    for i in range(params.num_client):
        data_idx = client_idxlist[i]
        dataloader = DataLoader(DatasetSplit(dataset,data_idx),batch_size=params.client_bs,shuffle=True)
        # ds = [dataset[idx] for idx in data_idx]
        # dataloader = DataLoader(ds,batch_size=params.client_bs,shuffle=True)
        all_loader.append(dataloader)
        client_samp_idxs[i] = data_idx
    
    return all_loader


def iid(dataset:datasets,params):
    '''
    Sample I.I.D. client data from MNIST dataset
    param dataset:
    param num_users:
    param batch_size: client
    return: list of DataLoader 
    '''
    # ll = [1]
    # print(len(ll))
    num_clients = params.num_client
    batch_size = params.client_bs

    num_items = int(len(dataset)/ num_clients)
    all_idxs = list(range(len(dataset)))
    all_loaders = {}
    for i in tqdm(range(num_clients)):
        samp_idxs = np.random.choice(all_idxs,num_items,replace=False) #False 表示不可以取相同数字 
        img = []
        label = []
        for idx in samp_idxs:
            img.append(dataset[idx][0])
            label.append(torch.tensor(dataset[idx][1]))

        img = torch.stack(img)
        label = torch.stack(label)
        ds = TensorDataset(img,label)
        dataloader = DataLoader(ds,batch_size=batch_size,shuffle=True)
        all_loaders[i]=dataloader
        all_idxs = list(set(all_idxs)-set(samp_idxs))
    
    return all_loaders

# def non_iid(dataset, num_user):
#     pass


def noniid(dataset:datasets,params):
    '''
    使用dirichlet distribution来进行采样，模仿非独立同分布
    参考 How To Backdoor Federated Learning
        【AAAI】BEAS: Blockchain Enabled Asynchronous & Secure Federated Machine Learning
    time:2022/2/11
    by:zhenyuan guo
    ''' 
    all_loader = {}
    dataset_class={}
    batch_size = params.client_bs
    for idx, (data,target) in enumerate(dataset):
        if target in dataset_class.keys():
            dataset_class[target].append(idx)
        else:
            dataset_class[target]=[idx]
    
    class_size = [len(i) for i in dataset_class.values()]
    class_nomb = len(dataset_class.keys())
    client_idxlist = defaultdict(list)

    for i in range(class_nomb):
        random.shuffle(dataset_class[i])
        sample_numb = class_size[i] * np.random.dirichlet(np.array(params.num_client * [params.noniid_alpha]) )
        for client in range(params.num_client):
            data_nomb = int(round(sample_numb[client]))
            sampled_list = dataset_class[i][:min(class_size[i],data_nomb)]
            client_idxlist[client].extend(sampled_list)
            dataset_class[i] = dataset_class[i][min(class_size[i],data_nomb):]
    
    for i in range(params.num_client):
        data_idx = client_idxlist[i]

        img = []
        label = []
        for idx in data_idx:
            img.append(dataset[idx][0])
            label.append(torch.tensor(dataset[idx][1]))
        img = torch.stack(img)
        label = torch.stack(label)
        ds = TensorDataset(img,label)
        dataloader = DataLoader(ds,batch_size=batch_size,shuffle=True)
        # dataloader = DataLoader(DatasetSplit(dataset,data_idx),batch_size=params.client_bs,shuffle=True,num_workers=6)
        # ds = [dataset[idx] for idx in data_idx]
        # dataloader = DataLoader(ds,batch_size=params.client_bs,shuffle=True)
        all_loader[i]=dataloader
    
    return all_loader


def root_data(dataset:datasets,params):
    dataset_class={}
    # 将每个标签数据索引构建成一个字典
    for idx, (data,target) in enumerate(dataset):
        if target in dataset_class.keys():
            dataset_class[target].append(idx)
        else:
            dataset_class[target]=[idx]
    
    sample_idx = []
    for label, idx in dataset_class.items():
        random.shuffle(idx)
        sample_idx.extend(idx[:params.onud_root_dataset_numb])
    
    ds = [dataset[idx] for idx in sample_idx]
    dataloader = DataLoader(ds,batch_size=params.client_bs,shuffle=True,num_workers=2)

    return dataloader
    



'''返回的是每个客户机都是相同数量的noniid数据'''
def noniid_equal(dataset:datasets,params):
    num_clients = params.num_client
    batch_size = params.client_bs
    k_shard = 2 # 每个客户机片数

    num_shards = int(k_shard * num_clients) #划分成片数
    num_shards_datas = int(len(dataset) / num_shards) #每一片有多少数据

    idxs_shard = [i for i in range(num_shards)] # 每片的索引

    # 将target进行排序
    idxs_target_sorted = dataset.targets.numpy().argsort()

    all_loaders = []

    for i in range(num_clients):
        samp_shard_idxs = np.random.choice(idxs_shard,k_shard,replace=False) # 随机从没有选过的片中选取片数
        tmp_data_idxs = []
        for samp_shard_idx in samp_shard_idxs:
            start_idx = (samp_shard_idx-1)*num_shards_datas #开始的suoyin
            end_idx = (samp_shard_idx)*num_shards_datas
            tmp_idx = [tmp for tmp in range(start_idx,end_idx)]
            tmp_data_idxs = tmp_data_idxs + tmp_idx
        
        samp_idxs = [idxs_target_sorted[tmp_data_idx] for tmp_data_idx in tmp_data_idxs]
        ds = [dataset[samp_idx] for samp_idx in samp_idxs] 
        dataloader = DataLoader(ds,batch_size=batch_size,shuffle=True)
        all_loaders.append(dataloader)
        idxs_shard = list(set(idxs_shard)-set(samp_shard_idxs))
    
    return all_loaders

def noniid_unequal(dataset:datasets,params):

    num_clients = params.num_client
    batch_size = params.client_bs

    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_shards_datas = 240, int(len(dataset)/240)
    min_shard, max_shard = 2,30

    idxs_shard = [i for i in range(num_shards)] # 每片的索引

    # 将target进行排序
    idxs_target_sorted = dataset.targets.numpy().argsort()

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,size=num_clients)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)
    random_shard_size.sort()
    random_shard_size[-1] = random_shard_size[-1] - (sum(random_shard_size)-num_shards)

    all_loaders = []
    for i in range(num_clients):
        samp_shard_idxs = np.random.choice(idxs_shard,random_shard_size[i],replace=False) # 随机从没有选过的片中选取片数
        tmp_data_idxs = []
        for samp_shard_idx in samp_shard_idxs:
            start_idx = (samp_shard_idx-1)*num_shards_datas #开始的suoyin
            end_idx = (samp_shard_idx)*num_shards_datas
            tmp_idx = [tmp for tmp in range(start_idx,end_idx)]
            tmp_data_idxs = tmp_data_idxs + tmp_idx
        
        samp_idxs = [idxs_target_sorted[tmp_data_idx] for tmp_data_idx in tmp_data_idxs]
        ds = [dataset[samp_idx] for samp_idx in samp_idxs] 
        dataloader = DataLoader(ds,batch_size=batch_size,shuffle=True)
        all_loaders.append(dataloader)
        idxs_shard = list(set(idxs_shard)-set(samp_shard_idxs))    

    return all_loaders   

    





