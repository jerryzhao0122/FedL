# -*- coding: utf-8 -*-
'''
time:2022/1/24
author:Guo Zhenyuan
'''

import os
import copy
import time
import numpy as np
import torch 
import warnings

from tensorboardX import SummaryWriter

from configs.fedl_params import fl_params
from configs.model_params import ml_params
from fedlearn.client import Attacker, Clients
from fedlearn.server import Server  
from model.cnn import CNNCifar, Mnist
from utils.data import get_dataset
from utils.sampling import iid,noniid_equal,noniid_unequal,noniid, root_data
from torch.utils.data import DataLoader
from model.alexnet import AlexNet
from model.resnet import ResNet

warnings.filterwarnings('ignore')

# def acc(dataset,model):
    
#     model.eval()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     correct = 0
#     total = 0
#     for data,target in dataset:
#         outputs = model(data)
#         _, pred_labels = torch.max(outputs,1)
#         pred_labels = pred_labels.view(-1)
#         correct += torch.sum(torch.eq(pred_labels, target)).item()
#         total += len(target)
#     return correct

if __name__ == '__main__':
    strat_time = time.time()
    
    # define paths
    # path_project = os.path.abspath('..')
    # logger = SummaryWriter('logs')

    fl_p = fl_params()
    # ml_p = ml_params()

    # path_project = os.path.abspath('..')
    # writer = SummaryWriter('logs')

    # 获取数据
    train_dataset, test_dataset = get_dataset(fl_p)
    root_dataloader = root_data(train_dataset,fl_p)
    print('获取数据完毕')

    if fl_p.white_box == True:
        print('White Box Get Model')
        train_dataset.data = train_dataset.data[:30000]
        train_dataset.targets = train_dataset.targets[:30000]


    #训练太慢，因此获取1/5的数据
    # if fl_p.rsa == True:
    #     train_dataset.data = train_dataset.data[:12000]
    #     train_dataset.targets = train_dataset.targets[:12000]


    if fl_p.data_distributed == 'iid':
        train_loaders = iid(train_dataset,fl_p)
    elif fl_p.data_distributed == 'noniid_equal':
        train_loaders = noniid_equal(train_dataset,fl_p)
    elif fl_p.data_distributed == 'noniid_unequal':
        train_loaders = noniid_unequal(train_dataset,fl_p)
    elif fl_p.data_distributed == 'noniid':
        train_loaders = noniid(train_dataset,fl_p)
    else:
        print('请输入正确的数据分布类型')
    
    print('dataloaer划分完毕')
    test_loader = DataLoader(test_dataset,batch_size=fl_p.client_bs,shuffle=True)


    # for i,(data,target) in enumerate(train_loaders[0]):
    #     print(data)
    #     print(target)  
    #     break

    # 初始化模型
    # start_model = Mnist(fl_p).cuda()

    print('开始加载模型：',fl_p.model)
    if fl_p.model == 'resnet':
        start_model = ResNet().cuda()
    elif fl_p.model == 'lanet':
        start_model = CNNCifar(fl_p).cuda()
    elif fl_p.model == 'alexnet':
        start_model = AlexNet(num_classes=10).cuda()
    else:
        pass

    print('加载模型完毕')
    # for i,j in zip(start_model_client.state_dict().values(),start_model_server.state_dict().values()):
    #     print('ser:',i[0])
    #     print('cli:',j[0])
    #     break

    # 初始化Server
    server = Server(start_model,test_loader,fl_p,root_dataloader)

    # 初始化Clients
    clients = Clients(train_loaders,fl_p)
    

    #存在攻击
    if fl_p.attack == True:
        attacker_num = fl_p.attack_num
        
        # 解决non-iid 时候标签反转攻击中不存在标签的情况
        if (fl_p.attack_type == 'label') and (fl_p.data_distributed == 'noniid'):
            attacker_ids = []
            have_label_client_num = 0
            for client in clients.all_clients.values():
                cl_dataloader = client.dataloader
                for d,l in cl_dataloader:
                    l_list = list(l)
                    if fl_p.labelflip_original_label in l_list:
                        have_label_client_num = have_label_client_num + 1
                        attacker_ids.append(client.id)
                        break
            # 如果拥有标签的数量大于攻击数量，就取前几个，如果小于攻击数量，就更改攻击的客户机数量
            if have_label_client_num > attacker_num:
                attacker_ids = attacker_ids[:attacker_num-1]
            elif have_label_client_num < attacker_num:
                fl_p.attack_num = have_label_client_num
            else:
                attacker_ids = attacker_ids

        else:  
            attacker_ids = [i for i in range(int(fl_p.attack_num))]

        for attacker_id in attacker_ids:
            attacker_data_loader = clients.all_clients[attacker_id].dataloader
            attacker = Attacker(attacker_id,attacker_data_loader,copy.deepcopy(start_model),1,fl_p) 
            clients.all_clients[attacker_id] = attacker


    for r in range(1,fl_p.round+1):
        print('Round: {} Start'.format(r))
        #进行联邦学习
        broadcast_weight = server.broadcast()
        update_weights = clients.update(start_model,copy.deepcopy(broadcast_weight))
        torch.cuda.empty_cache()
        server.aggregation(copy.deepcopy(update_weights))

        #学习的信息展示
        server.printinfo()

        #写入tensorboard
        server.writerLogs()
        print('Round: {} End'.format(r))
        # if r in [100,150,200,250,300]:
        #     print('Start Save model')
        #     server.saveModes()

    # server.saveInfo()
    print('okk--train---end')
    




