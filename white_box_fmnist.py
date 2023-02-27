from cProfile import label
import os
import glob
import time
import random
from turtle import forward
from cv2 import dnn_DetectionModel
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader  
import torchvision
import matplotlib.pyplot as plt
import shutil
import pickle
from PIL import Image # 8.0.1
import argparse
import subprocess
import wandb
from torchvision import transforms
from zmq import device

# selected_conv_layer_names = ['conv1','conv2','fc1','fc2','fc3']
selected_layer_names = ['layer1.0','layer2.0','layer3.0','layer4.0','layer5.0',
                            'fc1.1','fc2.1','fc3.1']
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50000, 
        help="""Only the first N samples of defender and reserve data will be used, 
        this means 2 * N samples in total.""")
    parser.add_argument("--input_feature_path", type=str, 
        default="/home/featurize/result/feature", 
        help="""where to load the input feature of the white-box attacker neural network.""")
    parser.add_argument("--save_attacker_model_path", type=str, 
        default="attacker_NN_model_checkpoints", 
        help="""where to save the checkpoints of the white-box attacker neural network.""")
    parser.add_argument('--random_seed', type=int, help='random seed', default=68)
    parser.add_argument('--lr', type=float, help='learning rate of the optimizer', default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--zip", action="store_true", default=False)
    parser.add_argument('--smp_flie',type=str,default='',help='采样文件名')
    parser.add_argument('--client_model_file',type=str,default='',help='client目标模型')
    parser.add_argument('--server_model_file',type=str,default='',help='server目标模型')
    parser.add_argument('--attack_server',type=bool,default=False)
    parser.add_argument('--attack_server_only',type=bool,default=False)

    parser.add_argument('--client_id',type=int,default=3)
    args = parser.parse_args()
    return args

def load_input_features(args):
    with open(os.path.join(args.input_feature_path, "L_all.pkl"), "rb") as f:
        L_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "hidden_all.pkl"), "rb") as f:
        hidden_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "gradients_all.pkl"), "rb") as f:
       gradients_all = pickle.load(f) 

    with open(os.path.join(args.input_feature_path, "y_all.npy"), "rb") as f:
        y_all = np.load(f)

    with open(os.path.join(args.input_feature_path, "yhat_all.npy"), "rb") as f:
        yhat_all = np.load(f)

    return L_all,hidden_all,gradients_all,y_all,yhat_all


# class NDataset(Dataset.Dataset):
#     def __init__(self,args,train):
#         with open(os.path.join(args.input_feature_path, "L_all.pkl"), "rb") as f:
#             L_all = pickle.load(f) 
#         with open(os.path.join(args.input_feature_path, "hidden_all.pkl"), "rb") as f:
#             hidden_all = pickle.load(f) 
#         with open(os.path.join(args.input_feature_path, "gradients_all.pkl"), "rb") as f:
#             gradients_all = pickle.load(f) 
#         with open(os.path.join(args.input_feature_path, "y_all.npy"), "rb") as f:
#             y_all = np.load(f)
#         with open(os.path.join(args.input_feature_path, "yhat_all.npy"), "rb") as f:
#             yhat_all = np.load(f)
#         self.L = torch.from_numpy(np.array(L_all)) 
#         self.hidden = hidden_all
#         self.grad = gradients_all
#         self.y  = torch.from_numpy(y_all)
#         self.yhat = torch.from_numpy(yhat_all)
#         label1 = torch.ones(30000).to(torch.int64)
#         label2 = torch.zeros(20000).to(torch.int64)
#         self.label = torch.cat([label1,label2],dim=0)
#         if train:
#             self.index = [i for i in range(15000,45000)]
#         else:
#             self.index = [i for i in range(0,5000)] + [i for i in range(45000,50000)]
    
#     def __len__(self):
#         return len(self.index)
    
#     def __getitem__(self,index):
#         l = self.L[self.index[index]]
#         h = self.hidden[self.index[index]]
#         g = self.grad[self.index[index]]
#         # h={'conv1':[],'conv2':[],'fc1':[],'fc2':[],'fc3':[]}
#         # g={'conv1':[],'conv2':[],'fc1':[],'fc2':[],'fc3':[]}
#         # for hidden,grad in zip(self.hidden,self.grad):
#         #     for ln in selected_conv_layer_names:
#         #         h[ln].append(torch.from_numpy(hidden[ln]))
#         #         g[ln].append(torch.from_numpy(grad[ln]))
#         yhat = self.yhat[self.index[index]]
#         y = self.y[self.index[index]]
#         label = self.label[self.index[index]]
#         return l,h,g,yhat,y,label

class CnnForFcnGrad(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.dim1 = input_shape[0]
        self.dim2 = input_shape[1]

        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(1,self.dim2),stride=(1,1))
        self.fc1 = nn.Linear(100*self.dim1,128)
        self.fc2 = nn.Linear(128,64)
    
    def forward(self,x):
        # x = F.leaky_relu(self.conv1(self.dropout(x)))
        x = F.leaky_relu(self.conv1(x))
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        # x = F.leaky_relu(self.fc1(self.dropout(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(self.dropout(x)))
        return x

class CnnForCnnGrad(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.dim1 = input_shape[0]
        self.dim2 = input_shape[1]
        self.dim3 = input_shape[2]
        self.dim4 = input_shape[3]

        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv3d(in_channels=self.dim1,out_channels=100,kernel_size=(1,self.dim3,self.dim4),stride=(1,1,1))
        self.fc1 = nn.Linear(100*self.dim2,128)
        self.fc2 = nn.Linear(128,64)
    
    def forward(self,x):
        # x = F.leaky_relu(self.conv1(self.dropout(x)))
        x = F.leaky_relu(self.conv1(x))
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        # x = F.leaky_relu(self.fc1(self.dropout(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(self.dropout(x)))
        return x

class CnnForCnnLayer(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.dim1 = input_shape[0]
        self.dim2 = input_shape[1]
        self.dim3 = input_shape[2]

        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(in_channels=self.dim1,out_channels=self.dim1,kernel_size=(self.dim2,self.dim3),stride=(1,1))
        # self.bn1 = nn.BatchNorm2d(self.dim1)
        self.fc1 = nn.Linear(self.dim1,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,64)
    def forward(self,x):
        # x = F.leaky_relu(self.conv1(self.dropout(x)))
        x = F.leaky_relu(self.conv1(x))
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = F.leaky_relu(self.fc1(self.dropout(x)))
        # x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(self.dropout(x)))
        x = F.leaky_relu(self.fc3(self.dropout(x)))
        x = F.leaky_relu(self.fc4(self.dropout(x)))
        return x

class FcnForModel(nn.Module):
    def __init__(self, dim_in=128, dim_out=64):
        super().__init__() 
        self.fc1 = nn.Linear(dim_in, 128)
        self.fc2 = nn.Linear(128, dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(self.dropout(x)))
        # x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(self.dropout(x)))
        return x

class FcnForEncoder(nn.Module):
    def __init__(self, dim_in, dim_out=2):
        super().__init__() 
        self.fc1 = nn.Linear(dim_in, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,dim_out)
        self.dropout = nn.Dropout(0.2)
        

    def forward(self, x):
        x = F.leaky_relu(self.fc1(self.dropout(x)))
        x = F.leaky_relu(self.fc2(self.dropout(x)))
        x = F.leaky_relu(self.fc3(self.dropout(x)))
        # print(x)
        # print(x.grad)
        x = F.leaky_relu(self.fc4(self.dropout(x)))
        # print(x)
        # print(x.grad)
        x = F.softmax(x)
        # print(x)
        return x

class WhiteBox(nn.Module):
    def __init__(self,hidden_layer_features_shape=None,param_gradients_shape=None):
        super().__init__()

        # print(hidden_layer_features_shape)
        # print(param_gradients_shape)
        # hidden层的特征提取
        # self.hidden_cnn1 = CnnForCnnLayer(hidden_layer_features_shape['layer1.0'])
        # self.hidden_cnn2 = CnnForCnnLayer(hidden_layer_features_shape['layer2.0'])
        # self.hidden_cnn3 = CnnForCnnLayer(hidden_layer_features_shape['layer3.0'])
        # self.hidden_cnn4 = CnnForCnnLayer(hidden_layer_features_shape['layer4.0'])
        # self.hidden_cnn5 = CnnForCnnLayer(hidden_layer_features_shape['layer5.0'])
        # self.hidden_fc1 = FcnForModel(dim_in = hidden_layer_features_shape['fc1.1'][0],dim_out = 64)
        # self.hidden_fc2 = FcnForModel(dim_in = hidden_layer_features_shape['fc2.1'][0],dim_out = 64)
        # self.hidden_fc3 = FcnForModel(dim_in = hidden_layer_features_shape['fc3.1'][0],dim_out = 64)
        self.hidden_cnn1 = CnnForCnnLayer((96,14,14))
        self.hidden_cnn2 = CnnForCnnLayer((256,7,7))
        self.hidden_cnn3 = CnnForCnnLayer((384,3,3))
        self.hidden_cnn4 = CnnForCnnLayer((384,3,3))
        self.hidden_cnn5 = CnnForCnnLayer((384,3,3))
        self.hidden_fc1 = FcnForModel(dim_in = 4096, dim_out = 64)
        self.hidden_fc2 = FcnForModel(dim_in = 4096, dim_out = 64)
        self.hidden_fc3 = FcnForModel(dim_in = 10, dim_out = 64)

        # grad层的特征提取
        # self.grad_cnn1 = CnnForCnnGrad(param_gradients_shape['layer1.0.weight'])
        # self.grad_cnn2 = CnnForCnnGrad(param_gradients_shape['layer2.0.weight'])
        # self.grad_cnn3 = CnnForCnnGrad(param_gradients_shape['layer3.0.weight'])
        # self.grad_cnn4 = CnnForCnnGrad(param_gradients_shape['layer4.0.weight'])
        # self.grad_cnn5 = CnnForCnnGrad(param_gradients_shape['layer5.0.weight'])
        # self.grad_cnn6 = CnnForFcnGrad(param_gradients_shape['fc1.1.weight'])
        # self.grad_cnn7 = CnnForFcnGrad(param_gradients_shape['fc2.1.weight'])
        # self.grad_cnn8 = CnnForFcnGrad(param_gradients_shape['fc3.1.weight'])

        # 其他层的特征提取
        # self.yhat_fc = FcnForModel(dim_in = 10,dim_out = 64)
        # self.label_fc = FcnForModel(dim_in = 10,dim_out = 64)
        # self.loss_fc = FcnForModel(dim_in = 1,dim_out = 64)

        # encoder分类层
        self.encoder = FcnForEncoder(dim_in=64)

    
    def forward(self,h):
        
        # hidden层的特征提取向前传播   
        # print(h['conv1'].shape)
        # print(h['conv2'].shape)
        # print(h['fc1'].shape)
        # print(h['fc2'].shape)
        # print(h['fc3'].shape)

        # x_hidden_cnn1 = self.hidden_cnn1(h['layer1.0'])
        # x_hidden_cnn2 = self.hidden_cnn2(h['layer2.0'])
        # x_hidden_cnn3 = self.hidden_cnn3(h['layer3.0'])
        # x_hidden_cnn4 = self.hidden_cnn4(h['layer4.0'])
        # x_hidden_cnn5 = self.hidden_cnn5(h['layer5.0'])
        # x_hidden_fc1 = self.hidden_fc1(h['fc1.1'])
        # x_hidden_fc2 = self.hidden_fc2(h['fc2.1'])
        # x_hidden_fc3 = self.hidden_fc3(h['fc3.1'])
        x = self.hidden_fc2(h['fc2.1'])+self.hidden_fc3(h['fc3.1'])

        # x_cnn = self.hidden_cnn1(h['layer1.0']) + self.hidden_cnn2(h['layer2.0']) + self.hidden_cnn3(h['layer3.0'])+self.hidden_cnn4(h['layer4.0'])+self.hidden_cnn5(h['layer5.0'])
        # x_fc = self.hidden_fc1(h['fc1.1'])+ self.hidden_fc2(h['fc2.1'])+self.hidden_fc3(h['fc3.1'])
        # x = torch.cat([x_hidden_cnn1,x_hidden_cnn2,x_hidden_cnn3,x_hidden_cnn4,x_hidden_cnn5,x_hidden_fc1,x_hidden_fc2,x_hidden_fc3,],dim=1)
        # x = torch.cat([x_hidden_fc2,x_hidden_fc3],dim=1)
        # grad层向前传播
        # print(g['conv1'].shape)
        # print(g['conv2'].shape)
        # print(g['fc1'].shape)
        # print(g['fc2'].shape)
        # print(g['fc3'].shape)
        # x_grad_cnn1 = self.grad_cnn1(g['layer1.0.weight'])
        # x_grad_cnn2 = self.grad_cnn2(g['layer2.0.weight'])
        # x_grad_cnn3 = self.grad_cnn3(g['layer3.0.weight'])
        # x_grad_cnn4 = self.grad_cnn4(g['layer4.0.weight'])
        # x_grad_cnn5 = self.grad_cnn5(g['layer5.0.weight'])
        # x_grad_cnn6 = self.grad_cnn6(g['layer6.0.weight'])
        # x_grad_cnn7 = self.grad_cnn7(g['layer7.0.weight'])
        # x_grad_cnn8 = self.grad_cnn8(g['layer8.0.weight'])

        # 其他层向前传播
        # print(yhat.shape)
        # print(y.shape)
        # print(L.shape)
        # print(L)
        # print(yhat)
        # x_yhat_fc = self.yhat_fc(yhat)
        # x_label_fc = self.label_fc(y)
        # x_loss_fc = self.loss_fc(L)

        #拼接后传入encoder
        # print(x_hidden_cnn1.shape)
        # print(x_hidden_cnn2.shape)
        # print(x_hidden_fc1.shape)
        # print(x_hidden_fc2.shape)
        # print(x_hidden_fc3.shape)
        # print(x_grad_cnn1.shape)
        # print(x_grad_cnn2.shape)
        # print(x_grad_cnn3.shape)
        # print(x_grad_cnn4.shape)
        # print(x_grad_cnn5.shape)
        # print(x_yhat_fc.shape)
        # print(x_label_fc.shape)
        # print(x_loss_fc.shape)
        # x_hidden_cnn1,x_hidden_cnn2,x_hidden_fc1,x_hidden_fc2,x_hidden_fc3,
        # x = torch.cat([ x_hidden_cnn1,x_hidden_cnn2,x_hidden_cnn3,x_hidden_cnn4,x_hidden_cnn5,x_hidden_fc1,x_hidden_fc2,x_hidden_fc3,
        #                 x_grad_cnn1,x_grad_cnn2,x_grad_cnn3,x_grad_cnn4,x_grad_cnn5,x_grad_cnn6,x_grad_cnn6,x_grad_cnn7,
        #                 x_yhat_fc],dim=1)
        # print(x.shape)

        x = self.encoder(x)
        return x

class AlexNet_OneChannel(torch.nn.Module):
    def __init__(self, num_classes=10,track=True, init_weights=False):
        super(AlexNet_OneChannel, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 96, kernel_size=5, stride=2, padding=2),
                                          # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
                                          torch.nn.BatchNorm2d(96,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                          torch.nn.BatchNorm2d(256,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True))
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True))
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(384,track_running_stats=track),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(384, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(inplace=True))
        self.fc3 = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def load_target_model(model_path, device):
    model = AlexNet_OneChannel(track=False).to(device)
    model_params = torch.load(model_path,map_location=device)

    for key, value in model_params.items():
        if 'running_var' in key: # 保证方差大于0
            print(key)  
            model_params[key] = value.where(value<0,torch.zeros_like(value),value)
            print(model_params[key])

    model.load_state_dict(torch.load(model_path,map_location=device))
    return model

def load_attack_model(device):
    model = WhiteBox().to(device)
    return model

def load_dataloader(file,id):
    
    transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
    trained_dataset = torchvision.datasets.FashionMNIST(
            root="/home/featurize/data",
            train=True,
            download=True,
            transform=transform
        )
    
    untrained_dataset = torchvision.datasets.FashionMNIST(
            root="/home/featurize/data",
            train=False,
            download=True,
            transform=transform
        )
    
    # with open(os.path.join('/home/featurize/result/models/client', "iid_P_FedAvg_CDP_15_client_samp_idxs.pkl"), "rb") as f:
    #         clients_samp_idx = pickle.load(f) 
    with open(file,'rb') as f:
        clients_samp_idx = pickle.load(f)
    # clients_samp_idx=[0]
    # clients_samp_idx[0] = list(range(10000))

    label_dict_idx1 = {}
    for idx,label in enumerate(trained_dataset.train_labels.tolist()):
        if idx in clients_samp_idx[id][:10000]:
            if label in label_dict_idx1.keys():
                label_dict_idx1[label].append(idx)
            else:
                label_dict_idx1[label] = [idx]

    idx1_1 = []
    idx1_2 = []
    for label,idxs in label_dict_idx1.items():
        idx1_1 = idx1_1 + idxs[:int(len(idxs)*0.8)] 
        idx1_2 = idx1_2 + idxs[int(len(idxs)*0.8):] 

    random.shuffle(idx1_1)
    random.shuffle(idx1_2)

    idx1 = idx1_1+idx1_2


    label_dict_dix2 = {}
    for idx,label in enumerate(untrained_dataset.train_labels.tolist()):
        if label in label_dict_dix2.keys():
            label_dict_dix2[label].append(idx)
        else:
            label_dict_dix2[label] = [idx]
    idx2_1 = []
    idx2_2 = []
    for label,idxs in label_dict_dix2.items():
        idx2_1 = idx2_1 + idxs[:int(len(idxs)*0.8)] 
        idx2_2 = idx2_2 + idxs[int(len(idxs)*0.8):] 

    random.shuffle(idx2_1)
    random.shuffle(idx2_2)

    idx2 = idx2_1+idx2_2

    client0_dataset_train = MkFlDataset(trained_dataset,untrained_dataset,clients_samp_idx[id],train=True,idx1=idx1,idx2=idx2)
    client0_dataset_test = MkFlDataset(trained_dataset,untrained_dataset,clients_samp_idx[id],train=False,idx1=idx1,idx2=idx2)

    train_dl = DataLoader(client0_dataset_train,batch_size=1024,num_workers=6,shuffle=True)
    test_dl = DataLoader(client0_dataset_test,batch_size=1024,num_workers=6,shuffle=True)

    # dataset.targets = [1]*30000 + [0]*20000

    # samp_idx = [i for i in range(10000,50000)]
    # train_dl = DataLoader(DatasetSplit(dataset,samp_idx),batch_size=128,num_workers=6,shuffle=True)
    # train_dl = DataLoader(dataset,batch_size=128,num_workers=6,shuffle=True)

    return train_dl,test_dl

def train(train_dl,test_dl,target_model,attack_model,device,epoch):

    hidden_layer_features = {}
    def get_hidden_layer_features(name):
        def fhook(model, input, output):
            hidden_layer_features[name] = output
            # print('forward',name,output.shape)
        return fhook

    # optim = torch.optim.SGD(attack_model.parameters(), lr=0.005,momentum=0.9)
    optim = torch.optim.Adamax(attack_model.parameters(), lr=0.001)
    # optim = torch.optim.Adam(attack_model.parameters(),lr=0.002)
    criterion = torch.nn.CrossEntropyLoss()

    # 首先对hook函数进行注册
    for name, layer in target_model.named_modules():
        if name in selected_layer_names:
            if isinstance(layer,torch.nn.modules.conv.Conv2d):
                layer.register_forward_hook(get_hidden_layer_features(name))
            if isinstance(layer,torch.nn.modules.linear.Linear):
                layer.register_forward_hook(get_hidden_layer_features(name))

    attack_model.train()
    target_model.eval()
    best_acc=0
    for e in range(epoch):

        t0 = time.time()

        train_loss = 0
        test_loss = 0
        train_correct_cnt = 0
        test_correct_cnt = 0
        train_total_cnt = 0
        test_total_cnt = 0
        

        attack_model.train()
        for batch,(images, labels) in enumerate(train_dl):
            # print(images)
            # print(labels)
            images, labels = images.to(device), labels.to(device)

            # 根据target_model获取隐藏层的特征

            # 使用前向传播获取隐藏层输出
            output1 = target_model(images)

            # 使用开始训练attack模型
            output2 = attack_model(hidden_layer_features)
            loss = criterion(output2,labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # print(output)
            # print(output.argmax(dim=1))
            # print(ll)

            train_loss += loss.item()
            train_correct_cnt += (output2.argmax(dim=1) == labels).sum().item() 
            train_total_cnt += len(labels)

            # acc = acc + (output.argmax(dim=1)==ll).sum().item()
            # print(loss.item())
            # print(acc)
        
        attack_model.eval()
        for batch,(images, labels) in enumerate(test_dl):
            # print(images)
            # print(labels)
            images, labels = images.to(device), labels.to(device)

            # 根据target_model获取隐藏层的特征

            # 使用前向传播获取隐藏层输出
            output1 = target_model(images)

            # 使用开始训练attack模型
            output2 = attack_model(hidden_layer_features)

            loss = criterion(output2,labels)

            test_loss += loss.item()
            test_correct_cnt += (output2.argmax(dim=1) == labels).sum().item() 
            test_total_cnt += len(labels)


        train_loss /= len(train_dl)
        train_acc = (train_correct_cnt / train_total_cnt) * 100
        test_loss /= len(test_dl)
        test_acc = (test_correct_cnt / test_total_cnt) * 100

        t1 = time.time() - t0

        print("Epoch {} | Train loss {:.5f} | Train acc {:.2f} | Test loss {:.5f} | Test acc {:.2f} | Time {:.1f} seconds.".format(
                e+1, train_loss, train_acc, test_loss, test_acc, t1))
        
        if best_acc<test_acc:
            best_acc=test_acc
    print('best test acc: {}'.format(best_acc))
    # wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "train_epoch_time": t1})


class MkFlDataset(Dataset):
    def __init__(self,dataset_trained,dataset_notrained,samp_idxs,train=True,idx1=[],idx2=[]):
        # print(samp_idxs)

        if idx1!=[]:
            self.idx1=idx1
        else:
            if len(samp_idxs)>10000:
                self.idx1 = samp_idxs[:10000]
            self.idx1 = [int(i) for i in self.idx1]

        if idx2!=[]:
            self.idx2=idx2
        else:
            self.idx2 = [int(i) for i in range(len(self.idx1))]
        # print(self.idx2)
        self.dataset_trained = dataset_trained
        self.dataset_untrained = dataset_notrained

        if train == True:
            self.idx1 = self.idx1[:int(len(self.idx1)*0.8)]
            self.idx2 = self.idx2[:int(len(self.idx2)*0.8)]
            # self.idx1 = list(range(15000))
            # self.idx2 = list(range(5000))
        else:
            self.idx1 = self.idx1[int(len(self.idx1)*0.8):]
            self.idx2 = self.idx2[int(len(self.idx2)*0.8):]
            # self.idx1 = list(range(15000,20000))
            # self.idx2 = list(range(5000,10000))

        self.lenidxs = len(self.idx1) + len(self.idx2)
    
    def __len__(self):
        return self.lenidxs
    
    def __getitem__(self, index):
        if index < len(self.idx1): # 说明是经过训练的数据
            # print(self.idx1[index])
            image,_ = self.dataset_trained[self.idx1[index]]
            label = torch.tensor(1)
        else: # 说明是没经过训练的数据
            image,_ = self.dataset_untrained[self.idx2[index-len(self.idx1)]]
            label = torch.tensor(0)
        
        return image,label


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(self.dataset[self.idxs[item]])
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

if __name__ == "__main__":

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for PyTorch")
    else:
        device = torch.device("cuda")
        print("Using GPU for PyTorch") 

    args = parse_arguments()

    client_model_path = '/home/featurize/result/models/client/' + args.client_model_file
    server_model_path = '/home/featurize/result/models/server/' + args.server_model_file
    smp_file_path = '/home/featurize/result/models/client/' + args.smp_flie

    print('smpleing file is {}'.format(args.smp_flie))

    # for i in range(4):
    # 获取数据
    i = args.client_id
    train_dl,test_dl = load_dataloader(smp_file_path,id=i)
    print('client:{} 数据获取完毕'.format(i))

    # 训练client attack model
    print('目标模型为客户端上传的模型参数')
    # 加载目标模型
    client_target_model = load_target_model(client_model_path,device)
    client_attack_model = load_attack_model(device)
    # client_attack_model.apply(weights_init)
    print('目标模型和攻击模型加载完毕，开始target:client{}训练'.format(i))
    epoch = 200
    if args.attack_server_only != True:
        train(train_dl,test_dl,client_target_model,client_attack_model,device,epoch)
    print('target_client attack finsh !!! \n\n')

    if args.attack_server:
        # 训练server attack model
        print('目标模型为服务器广播的模型参数')
        # 加载目标模型
        server_target_model = load_target_model(server_model_path,device)
        server_attack_model = load_attack_model(device)
        # server_attack_model.apply(weights_init)
        print('目标模型和攻击模型加载完毕，开始target:server训练')
        epoch = 200
        train(train_dl,test_dl,server_target_model,server_attack_model,device,epoch)

    # # 获取目标模型和攻击模型
    # model_path = '/home/featurize/result/models/client/iid_P_FedAvg_CDP_15_client3_round250.pth.tar'
    # # iidPrivacyFlFedSIGNclient0_round300.pth.tar
    # target_model = load_target_model(model_path,device)
    # attack_model = load_attack_model(device)
    # print('目标模型攻击模型加载完毕')

    # # 开始训练模型
    # epoch=1000
    # train(train_dl,test_dl,target_model,attack_model,device,epoch)
