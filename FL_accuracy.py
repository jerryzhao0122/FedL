
'''
time: 2022/6/7
author: Guo Zhenyuan
'''

import os
import copy
import time
import numpy as np
import torch 
import warnings
import pickle

from tensorboardX import SummaryWriter
from model.cnn import CNNCifar, Mnist
from model.resnet import ResNet
from configs.fedl_params import fl_params
from fedlearn.client import Clients
from fedlearn.server import Server  
from utils.data import get_dataset
from utils.sampling import iid, noniid,whitebox_iid
from torch.utils.data import DataLoader
from model.alexnet import AlexNet

warnings.filterwarnings('ignore')

def save_pickle_object(results_dir, file_name, obj):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    print("{} saved.".format(file_path))

if __name__ == '__main__':
    
    # 加载参数
    fl_p = fl_params()

    # 获取数据
    train_dataset, test_dataset = get_dataset(fl_p)
    print("数据获取完毕")

    # if fl_p.white_box == True:
    #     print('执行白盒攻击，保存客户机的模型')
    #     # 将数据进行划分
    #     train_dataset.data = train_dataset.data[:30000]
    #     train_dataset.targets = train_dataset.targets[:30000]

    if fl_p.data_distributed == 'iid':
        train_loaders = iid(train_dataset,fl_p)
    elif fl_p.data_distributed == 'noniid':
        train_loaders = noniid(train_dataset,fl_p)
    else:
        print('请输入正确的数据分布类型')

    # keyfind = client_samp_idxs[0][0]
    # print('keyfind',keyfind)

    print('dataloader划分完毕')
    test_loader= DataLoader(test_dataset,batch_size=fl_p.client_bs,shuffle=True)

    print('开始加载模型：',fl_p.model)
    if fl_p.model == 'resnet':
        start_model = ResNet().cuda()
    elif fl_p.model == 'lanet':
        if fl_p.dataset == 'cifar10':
            start_model = CNNCifar(fl_p).cuda()
        else:
            start_model = Mnist(fl_p).cuda()
    elif fl_p.model == 'alexnet':
        start_model = AlexNet(num_classes=10,track=False).cuda()
    else:
        pass
    print('加载模型完毕')

    # 初始化Server
    server = Server(start_model,test_loader,fl_p)

    # 初始化Clients
    clients = Clients(train_loaders,fl_p)

    for r in range(1,fl_p.round+1):
        print('Round: {} Start'.format(r))

        #进行联邦学习
        broadcast_weight = server.broadcast()
        update_weights = clients.update(broadcast_weight)
        server.aggregation(update_weights)

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
        