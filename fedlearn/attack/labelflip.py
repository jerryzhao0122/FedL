import torch
from torch.utils.data import DataLoader,Dataset,TensorDataset

'''获取反转后的数据和不需要反转的数据'''
def poisonLabelFlip(params,dataloader):
    all_data=[]
    all_target=[]

    for i,j in dataloader.dataset:
        if j == params.labelflip_original_label:
            j = params.labelflip_target_label
        j = torch.tensor(j)
        all_data.append(i)
        all_target.append(j)

    new_dataset_data = torch.stack(all_data)
    new_dataset_target = torch.stack(all_target)

    fliped_dataset = TensorDataset(new_dataset_data,new_dataset_target)
    fliped_loader = DataLoader(fliped_dataset,batch_size=params.client_bs,shuffle=True)
    
    return fliped_loader

'''只获取反转之后的数据'''
def getLabelFlip(params,dataloader):
    all_data=[]
    all_target=[]

    for i,j in dataloader.dataset:
        if j == params.labelflip_original_label:
            j = torch.tensor(params.labelflip_target_label)
            all_data.append(i)
            all_target.append(j)

    new_dataset_data = torch.stack(all_data)
    new_dataset_target = torch.stack(all_target)

    fliped_dataset = TensorDataset(new_dataset_data,new_dataset_target)
    fliped_loader = DataLoader(fliped_dataset,batch_size=params.client_bs,shuffle=True)
    
    return fliped_loader

def labelFlip(params,dataloader):
    all_data=[]
    all_target=[]

    for i,j in dataloader.dataset:
        all_data.append(i)
        # all_target.append(torch.tensor(10-j-1))
        all_target.append(torch.tensor(9-j))

    new_dataset_data = torch.stack(all_data)
    new_dataset_target = torch.stack(all_target)

    fliped_dataset = TensorDataset(new_dataset_data,new_dataset_target)
    fliped_loader = DataLoader(fliped_dataset,batch_size=params.client_bs,shuffle=True)
    
    return fliped_loader