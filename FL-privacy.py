
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
from configs.fedl_params import fl_params
from fedlearn.client import Clients
from fedlearn.server import Server  
from utils.data import get_dataset,get_dataset_wbox
from utils.sampling import noniid,whitebox_iid
from torch.utils.data import DataLoader
from model.alexnet import AlexNet,AlexNet_OneChannel
from model.cnn import  Mnist

warnings.filterwarnings('ignore')

# 可直接调用此函数
def set_seed(SEED=0):
  np.random.seed(SEED)
  torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
  torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
  torch.backends.cudnn.deterministic = True

def save_pickle_object(results_dir, file_name, obj):
    file_path = os.path.join(results_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    print("{} saved.".format(file_path))

if __name__ == '__main__':
    
    set_seed()
    # 加载参数
    fl_p = fl_params()

    # 获取数据
    train_dataset, test_dataset = get_dataset(fl_p)
    # if fl_p.dataset == 'mnist' or fl_p.dataset == 'fmnist':
    #     train_dataset, test_dataset = get_dataset_wbox(fl_p)

    print("数据获取完毕")

    # if fl_p.white_box == True:
    #     print('执行白盒攻击，保存客户机的模型')
    #     # 将数据进行划分
    #     train_dataset.data = train_dataset.data[:30000]
    #     train_dataset.targets = train_dataset.targets[:30000]

    if fl_p.data_distributed == 'iid':
        train_loaders,client_samp_idxs = whitebox_iid(train_dataset,fl_p)
    elif fl_p.data_distributed == 'noniid':
        train_loaders = noniid(train_dataset,fl_p)
    else:
        print('请输入正确的数据分布类型')

    # keyfind = client_samp_idxs[0][0]
    # print('keyfind',keyfind)

    # 保存采样结果用于训练
    file_dir = '/home/featurize/result/models/client'
    file_name =  str(fl_p.data_distributed) +'_'+ str(fl_p.name) +  '_client_samp_idxs.pkl'
    save_pickle_object(file_dir,file_name,client_samp_idxs)

    print('数据划分完毕')

    test_loader = DataLoader(test_dataset,batch_size=fl_p.client_bs,shuffle=True)

    # if fl_p.sign==True:
    #     tt = False
    # else:
    #     tt = True
    tt = False
    start_model = AlexNet(track=tt).cuda()
    if fl_p.dataset in ['mnist','fmnist'] and fl_p.model == 'alexnet':
        start_model = AlexNet_OneChannel(track=tt).cuda()

    if fl_p.model == 'lanet':
        start_model = Mnist(fl_p).cuda()
    print('加载模型完毕')



    if fl_p.use_pretrain == True:
        start_model.load_state_dict(torch.load('/home/featurize/result/models/server/iid_P_FedAVG_1.pth.tar'))
        print('使用预训练模型')

    # 初始化Server
    server = Server(start_model,test_loader,fl_p)
    # 初始化Clients
    clients = Clients(train_loaders,fl_p)

    for r in range(1,fl_p.round+2):
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
        # if r in [100,200,300]:
        #     print('Start Save model')
        #     server.saveModes()

    # server.saveInfo()
    print('okk--train---end')
        