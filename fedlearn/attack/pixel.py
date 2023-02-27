
import torch
from torch.utils.data import DataLoader,Dataset,TensorDataset

def poisonPixeled(params,dataloader):
    all_data=[]
    all_target=[]
    p1 = params.pixel1
    p2 = params.pixel2
    for i,j in dataloader.dataset:
        i[:,-p1:,-p2:]=1
        j=torch.tensor(1)
        all_data.append(i)
        all_target.append(j)

    new_dataset_data = torch.stack(all_data)
    new_dataset_target = torch.stack(all_target)

    pixel_dataset = TensorDataset(new_dataset_data,new_dataset_target)
    pixel_loader = DataLoader(pixel_dataset,batch_size=params.client_bs,shuffle=True)
    
    return pixel_loader

