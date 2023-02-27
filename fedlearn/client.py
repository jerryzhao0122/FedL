'''
time: 2022/1/24
author: Guo Zhenyuan
'''

import sys
from math import gamma
from operator import sub

from torch.nn.modules import module

sys.path.append('..')

import copy

import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from model.alexnet import AlexNet, get_mnist_alexnet,AlexNet_OneChannel
from model.cnn import CNNCifar, Mnist
from model.resnet import ResNet
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from tensorboardX import SummaryWriter
from torch import nn

from .attack.labelflip import getLabelFlip, labelFlip, poisonLabelFlip
from .attack.pixel import poisonPixeled
from .attack.replace import dataReplace, getBackdoorDataloader
#package for computing individual gradients
# from backpack import extend
from .helper import (addWeight, bondsignWeight, flipWeight, l2Weight,
                     mulWeight, numMulWeight, process_grad_batch,
                     sameDircWeight, signWeight, sortDict, stosignWeight,
                     subWeight, sumWeight, weightToVec, zeroWeight)


class Clients():
    def __init__(self,dataloaders,params):
        self.params = params
        self.all_clients={}
        for id, dataloader in dataloaders.items():
            client = Client(id,dataloader,1,params)
            self.all_clients[id] = client
        # for id,dataloader in enumerate(dataloaders_list):
        #     client = Client(id,dataloader,1,params) # 开始时候round=1
        #     self.all_clients[id]=client
    
    def update(self,weight):
        '''
        return 一个字典{客户机id: 模型参数}，包含了当前round中选中的客户机
        '''
        update_weights = {}
        for c_id, client in self.all_clients.items():
            # print(weight['fc3.1.bias'])
            new_weight = client.update(copy.deepcopy(weight))
            update_weights[c_id] = new_weight
        
        # print(update_weights.keys())
        return update_weights

        # num_update_client = max(int(self.params.train_num),1)
        # ids_update_client = np.random.choice(range(self.params.num_client),num_update_client,replace=False)
        # update_weights = {}
        # for id in ids_update_client:
        #     update_client = self.all_clients[id]
        #     c_id = self.all_clients[id].id
        #     new_weight = update_client.update(copy.deepcopy(weight))
        #     # new_weight = update_client.update(weight)
        #     update_weights[c_id] = new_weight
        # return sortDict(update_weights)
        

class Client(object):
    def __init__(self,id,dataloader,now_round,params):
        '''
        客户机的基本信息
        '''
        self.id=id
        self.dataloader = dataloader
        # self.model = model
        self.now_round = now_round
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion=nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(model.parameters(),lr=self.params.client_lr,weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr, momentum=0.5)

        # onud中误差反馈的参数
        # self.onud_e = zeroWeight(self.model.state_dict())

        # server中相似参数
        # self.server_weight = zeroWeight(self.model.state_dict())

        path1 = 'logs/Client_ONUD_l2/' + str(self.id)
        self.writer = SummaryWriter(path1)

    def update(self,weight):
        # self.model.cuda()

        if self.params.update_type == 'weight':
            if self.params.ldp == True:
                new_weight = self.updateWeightWithLDP(weight)
            else:
                new_weight = self.updateWeight(weight)
        elif self.params.update_type == 'direction':

            if self.params.sign==True:
                new_weight = self.updateSIGN(weight)
            elif self.params.sign_v2==True:
                new_weight = self.updateSIGN_v2(weight)
            elif self.params.majority_vote==True:
                new_weight = self.updateMajorityVote(weight)
            elif self.params.dprsa==True:
                new_weight = self.updateDPRSA(weight)
            else:
                print('请输入正确的客户端更新方式')
        else:
            print('请输入正确的客户端更新方式')


        # if self.params.update_type == 'weight':
        #     print('weight')
        #     new_weight = self.updateWeight(weight)
        # elif self.params.update_type == 'dirction' and self.params.onud == True:
        #     print('you only need dirction')
        #     new_weight = self.updateONUD(weight)
        #     # new_weight = self.updateSIGN(weight)
        # elif self.params.update_type == 'dirction' and self.params.rsa == True:
        #     print('rsa, client update dirction')
        #     new_weight = self.updateWeight(weight)
        # elif self.params.sign == True:
        #     new_weight = self.updateSIGN(weight)
        # else:
        #     print('请输入正确的更新类型')


        self.now_round = self.now_round + 1
        return new_weight

    def updateDPRSA(self,weight):
        print('Client update using DPRSA ',self.params.dprsa_type)

        # model = AlexNet(track=False).cuda()
        if self.params.model == 'alexnet':
            # if self.params.sign:
            #     tt = False
            # else:
            #     tt = True
            tt = False
            model = AlexNet(track=tt).cuda()
            if self.params.dataset in ['mnist','fmnist']:
                model = AlexNet_OneChannel(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')

        model.load_state_dict(weight)

        # server的参数
        server_weight = copy.deepcopy(weight)
        
        model.train()
        
        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        # 设置优化器
        if self.params.white_box == True:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.rsa_alpha)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.rsa_alpha)
        
        self.client_epoch = 0

        for iter in range(self.params.client_ep):
            model.train()
            for batch_idx,(data,target) in enumerate(self.dataloader):
                # print(data.dtype)
                data,target = data.to(device),target.to(device)
                # print(data.shape)
                optimizer.zero_grad()
                log_probs = model(data)
                loss = criterion(log_probs,target)
                loss.backward()

                for p,sw in zip(model.parameters(),server_weight.values()):

                    if self.params.dprsa_type == "F":
                        # flip the sign information
                        random_flip = torch.rand_like(p,device=p.device)
                        flip = torch.ones_like(p,device=p.device)
                        flip[random_flip > self.params.dprsa_f_gamma] = -1
                        sign = torch.sign(sw-p)*flip
                    elif self.params.dprsa_type == 'G':
                        # add the Gaussian noise to the sign information
                        du = 2*0.01*0.01
                        epslion = self.params.dprsa_g_epslion
                        sigma = torch.maximum(2/3*(p.data - sw.data), 4*du/epslion * torch.ones_like(p))
                        gauss = torch.randn_like(p) * sigma
                        sign = torch.sign(sw-p+gauss)
                    else:
                        print('请输入正确的差分隐私类型')
                    p.grad.data = p.grad.data + self.params.rsa_l1_lambda * sign
                    # p.grad.data = p.grad.data

                optimizer.step()
            self.client_epoch = self.client_epoch + 1

        # 获取训练结果
        self.train_acc,self.train_loss = self.inference(model,self.dataloader)
        self.print_acc_loss()

        new_weight = model.state_dict()
        self.model = model

        if self.params.update_type == 'direction': #如果rsa只更新方向信息时候
            sign = signWeight(subWeight(server_weight,new_weight))
            return sign
        else:
            return new_weight
    
    def updateMajorityVote(self,weight):
        print('Client update use Majority vote')

        if self.params.model == 'alexnet':
            # if self.params.sign:
            #     tt = False
            # else:
            #     tt = True
            tt = False
            model = AlexNet(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')
        
        if self.now_round == 1:
            #  初始化动量
            self.v={}
            for name,value in model.named_parameters():
                self.v[name] = torch.zeros_like(value,device=value.device)

        model.load_state_dict(weight)

        model.train()

        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        self.client_epoch = 0

        for batch_idx,(data,target) in enumerate(self.dataloader):
            data,target = data.to(device),target.to(device)
            log_probs = model(data)
            loss = criterion(log_probs,target)
            loss.backward()
        
        sign_weight = {}
        for name, parms in model.named_parameters():
            sign_weight[name] = torch.sign(0.1*parms.grad + 0.9*self.v[name])

        # 更新动量
        for name, parms in model.named_parameters():
            self.v[name] = 0.1*parms.grad + 0.9*self.v[name]

        return sign_weight

    def updateSIGN_v2(self,weight):
        
        print('Client update using SIGN V2')
        # print('client start weight',weight['fc3.1.bias'])

        if self.params.model == 'alexnet':
            # if self.params.sign:
            #     tt = False
            # else:
            #     tt = True
            tt = False
            model = AlexNet(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')
        old_weight = copy.deepcopy(weight)
        model.load_state_dict(weight)

        model.train()

        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        self.client_epoch = 0

        # 优化器
        # optimizer = self.optimizer
        # optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        if self.params.white_box == True:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr,momentum=0.9)

        for iter in range(self.params.client_ep):
            model.train()
            for batch_idx,(data,target) in enumerate(self.dataloader):
                data,target = data.to(device),target.to(device)
                log_probs = model(data)
                loss = criterion(log_probs,target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.client_epoch = self.client_epoch + 1
        # 获取训练结果
        self.train_acc,self.train_loss = self.inference(model,self.dataloader)
        self.print_acc_loss()
        
        new_weight = model.state_dict()

        # print('client end weight',weight['fc3.1.bias'])
        # v2版本使用动量方法尝试    
        if self.now_round == 1:
            #  初始化动量
            self.v={}
            for name,value in model.named_parameters():
                self.v[name] = torch.zeros_like(value,device=value.device)
        
        # 根据动量计算需要传递的self.v = 0.1(new_weight-old_weight)+0.9self.v sign(self.v)
        self.v = addWeight(numMulWeight(0.1,subWeight(new_weight,old_weight)),numMulWeight(0.9,self.v))
        sign_weight = bondsignWeight(self.v,0.00000001)

        # direction = bondsignWeight(subWeight(new_weight,old_weight),0.00000001)
        # print(direction['fc3.1.bias'])

        return sign_weight

    
    def updateSIGN(self,weight):
        
        print('Client update using SIGN')
        # print('client start weight',weight['fc3.1.bias'])

        if self.params.model == 'alexnet':
            # if self.params.sign:
            #     tt = False
            # else:
            #     tt = True
            tt = False
            model = AlexNet(track=tt).cuda()
            if self.params.dataset in ['mnist','fmnist']:
                model = AlexNet_OneChannel(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')
        old_weight = copy.deepcopy(weight)
        model.load_state_dict(weight)

        model.train()

        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        self.client_epoch = 0

        # 优化器
        # optimizer = self.optimizer
        # optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        if self.params.white_box == True:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        else:
            # optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr,momentum=0.9)
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)

        for iter in range(self.params.client_ep):
            model.train()
            for batch_idx,(data,target) in enumerate(self.dataloader):
                data,target = data.to(device),target.to(device)
                log_probs = model(data)
                loss = criterion(log_probs,target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.client_epoch = self.client_epoch + 1
        # 获取训练结果
        self.train_acc,self.train_loss = self.inference(model,self.dataloader)
        self.print_acc_loss()
        
        new_weight = model.state_dict()

        # print('client end weight',weight['fc3.1.bias'])    
        direction = bondsignWeight(subWeight(new_weight,old_weight),0.00000001)
        # print(direction['fc3.1.bias'])

        return direction
    
    def updateONUD(self,dirct):
        '''
        author: ZhenyuanGuo
        date:2022/2/6
        需要一个跟新的方向矩阵
        返回的是一个梯度方向矩阵
        '''
        # print(dirct)
        # dirction,error = copy.deepcopy(dirct)
        dirction, server_weight_dirction,server_weight = dirct[0],dirct[1],dirct[2]
        # dirction = copy.deepcopy(dirct)  

        # old_weight = self.model.state_dict()

        # 不使用参数递减策略
        # self.lam = self.params.onud_client_lambda
        # self.lam = np.random.uniform(0.005,0.0002)

        # 使用参数递减策略防止损失上升
        # if self.now_round == 1:
        #     self.lam = self.params.onud_client_lambda
        # else:
        #     if self.now_round %10 == 0:
        #         self.lam = self.lam / 2.0

        # 使用正常方法 client_new_weight = client_old_weight - lambda * broadcast_weight
        # new_weight = subWeight(old_weight,numMulWeight(self.lam,dirction))
        # new_weight = copy.deepcopy(dirction)

        # 使用DP-LR方法
        # dp_lr = getDpLr(old_weight,self.lam,self.params.onud_sigma)
        # new_weight = subWeight(old_weight,mulWeight(dp_lr,dirction))
        
        # 使用every epoch + sign 方法
        # new_weight = copy.deepcopy(old_weight)

        # 使用误差修复方法
        # new_weight = subWeight(new_weight,numMulWeight(0.1*self.params.onud_client_lambda,error))
        
        # 尝试：当每次更新都加了一个聚合梯度值的时候会是什么样，这时候就不再先更新一次old weight
        # new_weight = copy.deepcopy(old_weight)

        # 客户机，服务器参数相同方向
        # new_weight = sameDircWeight(new_weight,server_weight_dirction)

        # 服务器方向修正
        # new_weight = addWeight(new_weight,numMulWeight(0.01,server_weight_dirction))

        # 直接使用服务器模型参数
        new_weight = server_weight
        
        tmp_weight = copy.deepcopy(new_weight)

        # model = self.model

        model = AlexNet().cuda()
        model.load_state_dict(new_weight)
        
        # 能够获取模型细节信息
        # model = extend(model)
        model.train()

        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        # 优化器
        # optimizer = self.optimizer
        optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)

        # Client Epoch  
        self.client_epoch = 0

        # 初始化方向weight
        # dirct_weight = zeroWeight(new_weight)

        for iter in range(self.params.client_ep):
            
            # 使用every epoch + sign 方法
            # tmp_weight = copy.deepcopy(model.state_dict())
            # tmp_weight = subWeight(tmp_weight,numMulWeight(self.lam,dirction))
            # model.load_state_dict(tmp_weight)

            batch_loss = []
            for batch_idx,(data,target) in enumerate(self.dataloader):
                data,target = data.to(device),target.to(device)
        
                log_probs = model(data)
                loss = criterion(log_probs,target)
                loss.backward()
                # for p,(key,value) in zip(model.parameters(),dirct_weight.items()):
                #     dirct_weight[key] = dirct_weight[key] + p.grad.data
                # for p,sw in zip(model.parameters(),dirct.values()):
                #     p.grad.data = p.grad.data - 0.01 * torch.sign(sw)
                    # p.grad.data = p.grad.data               
                # if iter%2==0:
                optimizer.step()
                optimizer.zero_grad()
            
            self.client_epoch = self.client_epoch + 1

            # 获取训练结果
            self.train_acc,self.train_loss = self.inference(model,self.dataloader)
            self.print_acc_loss()
        
        now_weight = copy.deepcopy(model.state_dict())
        # self.model.load_state_dict(tmp_weight)
        # 正常
        # self.model = model
        
        # pt = addWeight(subWeight(new_weight,now_weight),self.onud_e)
        # dirction = signWeight(pt)
        # self.onud_e = subWeight(pt, dirction)
        if self.params.onud == True:
            l2 = l2Weight(subWeight(tmp_weight,now_weight))
            self.writer.add_scalar('l2',l2, self.now_round)
        # dirction = signWeight(subWeight(new_weight,now_weight))
        dirction = bondsignWeight(subWeight(tmp_weight,now_weight),0.00000001)
        # error = signWeight(subWeight(subWeight(new_weight,now_weight),dirction))
        # dirction = stosignWeight(dirct_weight)
        # print(dirction)
        # torch.cuda.empty_cache()

        # dirction_and_error = [dirction,error]
        # print(dirction)
        return dirction
        # return (dirction,error)

    def updateWeightWithLDP(self,weight):
        # model = AlexNet(track=False).cuda()
        if self.params.model == 'alexnet':
            # if self.params.sign:
            #     tt = False
            # else:
            #     tt = True
            tt = False
            model = AlexNet(track=tt).cuda()
        if self.params.dataset in ['mnist','fmnist']:
                model = AlexNet_OneChannel(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')
        model.load_state_dict(weight)
        # model = extend(model)

        optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        criterion = nn.CrossEntropyLoss()

        sigma = get_noise_multiplier(
            target_delta=self.params.ldp_delta,
            target_epsilon=self.params.ldp_epsilon,
            # sample_rate=1/len(self.dataloader),
            sample_rate=1,
            steps=self.params.round,
            # epochs=self.params.client_ep,
            accountant='rdp'
        )
        print('Client update using Weight with LDP, smaple rate: {} epsilon: {} delta: {} sigma:{} clip: {}'.format(1/len(self.dataloader),self.params.ldp_epsilon,self.params.ldp_delta,sigma,self.params.ldp_clip))
        
        device = self.device
        self.client_epoch = 0

        for iter in range(self.params.client_ep):
            model.train()
            for i,(images, target) in enumerate(self.dataloader):
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)
                # compute output

                output = model(images)

                loss = criterion(output, target)
                loss.backward()

                # 计算所有梯度的l2范数
                g_norm_list = []
                for p in model.parameters():
                    g_norm_list.append(p.grad.data.view(-1))
                
                g_norm = torch.norm(torch.cat(g_norm_list) ,p=2)

                # 对梯度进行裁切添加噪声
                for p in model.parameters():
                    # p.grad.data = p.grad.data / max(1,g_norm/self.params.ldp_clip) + (1.0/self.params.client_bs)*torch.normal(0,sigma*self.params.ldp_clip,size=p.grad.data.shape,device=p.grad.device)
                    p.grad.data = p.grad.data / max(1,g_norm/self.params.ldp_clip) + 0.002*torch.normal(0,sigma*self.params.ldp_clip,size=p.grad.data.shape,device=p.grad.device)
    
                # with backpack(BatchGrad()):
                #     loss.backward()    
                #     process_grad_batch(list(model.parameters()),self.params.ldp_clip)
                #     for p in model.parameters():
                #         grad_noise = (1/self.params.client_bs)*torch.normal(0,sigma*self.params.ldp_clip,size=p.grad.shape,device=p.grad.device)
                #         p.grad.data = p.grad.data + grad_noise

                optimizer.step()
            
            self.client_epoch = self.client_epoch + 1
            # 获取训练结果
            self.train_acc,self.train_loss = self.inference(model,self.dataloader)
            self.print_acc_loss()
        
        new_weight = model.state_dict()
        del model
        
        return new_weight

    # def updateWeightWithLDP(self,weight):
        
    #     model = AlexNet(track=False).cuda()
    #     model.load_state_dict(weight)
    #     model = ModuleValidator.fix(model)
    #     ModuleValidator.validate(model,strict=False)

    #     optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr, momentum=0.9)
    #     criterion = nn.CrossEntropyLoss()

    #     privacy_engine = PrivacyEngine()

    #     model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #         module=model,
    #         optimizer=optimizer,
    #         data_loader=self.dataloader,
    #         epochs=self.params.client_ep,
    #         target_epsilon=self.params.ldp_epsilon,
    #         target_delta=self.params.ldp_delta,
    #         max_grad_norm=self.params.ldp_clip,
    #     )
        

    #     print('Client update using Weight with LDP, epsilon: {} delta: {} sigma:{} clip: {}'.format(self.params.ldp_epsilon,self.params.ldp_delta,optimizer.noise_multiplier,self.params.ldp_clip))

    #     device = self.device
    #     self.client_epoch = 0


    #     for iter in range(self.params.client_ep):
    #         model.train()
            
    #         with BatchMemoryManager(
    #             data_loader=train_loader,
    #             max_physical_batch_size=32,
    #             optimizer=optimizer
    #         ) as memory_safe_data_loader:
    #             for i,(images, target) in enumerate(memory_safe_data_loader):
    #                 optimizer.zero_grad()
    #                 images = images.to(device)
    #                 target = target.to(device)
                    
    #                 # compute output
    #                 output = model(images)

    #                 loss = criterion(output, target)
                        
    #                 loss.backward()
    #                 optimizer.step()

    #         self.client_epoch = self.client_epoch + 1
    #         # 获取训练结果
    #         self.train_acc,self.train_loss = self.inference(model,train_loader)
    #         self.print_acc_loss()
        
    #     new_weight = model.state_dict()
    #     return new_weight

    

    def updateWeight(self,weight):
        '''
        返回的是整个模型的情况
        '''
        # old_weight = copy.deepcopy(self.model.state_dict())
        # model = self.model

        print('Client update using Weight')

        # model = AlexNet(track=False).cuda()
        if self.params.model == 'alexnet':
            # if self.params.sign:
            #     tt = False
            # else:
            #     tt = True
            tt = False
            model = AlexNet(track=tt).cuda()
            if self.params.dataset in ['mnist','fmnist']:
                model = AlexNet_OneChannel(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')

        model.load_state_dict(weight)

        # server的参数
        server_weight = copy.deepcopy(weight)

        # 能够获取模型细节信息
        # model = extend(model)
        
        model.train()
        
        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        # 设置优化器
        if self.params.white_box == True:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr,momentum=0.5)
            # optimizer = torch.optim.Adamax(model.parameters(),lr=self.params.client_lr)
            # optimizer = torch.optim.Adam(model.parameters(),lr=self.params.client_lr)

        
        # if self.params.rsa == True:
        #     optimizer = torch.optim.SGD(model.parameters(),lr = self.params.alpha,momentum=0.9)
        # elif self.params.attack == True:
        #     optimizer = torch.optim.SGD(model.parameters(),lr=self.params.client_lr,momentum=0.5)
        #     # optimizer = torch.optim.Adamax(model.parameters(),lr=self.params.client_lr)
        #     # optimizer = torch.optim.SGD(model.parameters(),lr=self.params.lr,momentum=0.9)
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr)
        #     # optimizer = torch.optim.Adam(model.parameters(),lr=self.params.client_lr,weight_decay=0.1)

        # Client Epoch  
        self.client_epoch = 0

        for iter in range(self.params.client_ep):
            model.train()
            for batch_idx,(data,target) in enumerate(self.dataloader):
                # print(data.dtype)
                data,target = data.to(device),target.to(device)
                # print(data.shape)
                optimizer.zero_grad()
                log_probs = model(data)
                loss = criterion(log_probs,target)
                loss.backward()

                # if self.params.ldp == True: # 使用了dp-sgd
                #     pass
                # elif self.params.rsa == True: # 使用了rsa提高鲁棒性
                #     for p,sw in zip(model.parameters(),server_weight.values()):
                #         p.grad.data = p.grad.data + self.params.l1_lambda * torch.sign(sw-p)
                #         # p.grad.data = p.grad.data
        
                # else:
                #     # log_probs = model(data)
                #     # loss = criterion(log_probs,target)
                #     # loss.backward()
                #     pass
                optimizer.step()
                # batch_loss.append(loss.item())

            self.client_epoch = self.client_epoch + 1

        # 获取训练结果
        self.train_acc,self.train_loss = self.inference(model,self.dataloader)
        self.print_acc_loss()

        new_weight = model.state_dict()
        self.model = model

        if self.params.update_type == 'dirction': #如果rsa只更新方向信息时候
            sign = signWeight(subWeight(server_weight,new_weight))
            return sign

        return new_weight      

    def updateDelta(self,weight):
        delta_weight = subWeight(self.updateWeight(weight) - weight)
        return delta_weight
   

    def print_acc_loss(self):
        print('Client ID: {}\t Local Epoch: {}\t Acc: {:.5f}\t Loss: {:.5f}'.format(self.id,self.client_epoch,self.train_acc,self.train_loss))

    def inference(self,model,dataloader):
        model = model
        model.eval()
        loss,total,correct = 0.0,0.0,0.0
        device = self.device
        criterion = self.criterion
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += float(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


class Attacker(Client):
    def __init__(self, id, dataloader, now_round, params):
        super(Attacker,self).__init__(id, dataloader, now_round ,params)
        if params.attack_type == 'replace':
            self.malicious_dataloader = getBackdoorDataloader(self.params)
        elif params.attack_type == 'pixel':
            self.malicious_dataloader = poisonPixeled(self.params,self.dataloader)
        elif params.attack_type == 'label':
            # self.malicious_dataloader = poisonLabelFlip(self.params,self.dataloader)
            # self.malicious_dataloader = getLabelFlip(self.params,self.dataloader)
            self.malicious_dataloader = labelFlip(self.params,self.dataloader)
        elif params.attack_type == 'gaussian':
            pass
        else:
            pass


    def update(self,weight):
        
        # 如果是onud时候获取的是方向信息，因此要变换weight
        # if self.params.onud == True and self.params.attack_type == 'replace':
            # old_weight = copy.deepcopy(self.model.state_dict())
            # dirction = copy.deepcopy(weight)
            # weight = copy.deepcopy(subWeight(old_weight,numMulWeight(self.params.onud_client_lambda,dirction)))

        # if self.params.model == 'lanet':
        #     model = CNNCifar(self.params).cuda()
        # elif model == 'alexnet':
        #     model = AlexNet(track=False).cuda()
        # else:
        #     print('请输入正确的模型')
        if self.params.model == 'alexnet':
            tt = False
            model = AlexNet(track=tt).cuda()
        elif self.params.model == 'lanet':
            if self.params.dataset == 'cifar10':
                model = CNNCifar(self.params).cuda()
            else:
                model = Mnist(self.params).cuda()
        elif self.params.model == 'resnet':
            model = ResNet().cuda()
        else:
            print('请输入正确模型')

        if self.params.attack_type=='replace' and self.params.attack_replace_round <= self.now_round:
            '''进行模型替换的后门攻击，生成攻击的模型参数'''
            if self.params.onud == True:
                attacker_weight =  signWeight(subWeight(weight[2],self.updateReplace(model,weight[2])))
            elif self.params.sign == True:
                print('sign backdoor')
                # attacker_weight = signWeight(subWeight(weight,self.updateReplace(model,weight)))
                # attacker_weight =  signWeight(self.updateReplace(model,weight))
                attacker_weight = signWeight(subWeight(self.updateReplace(model,weight),weight))
            elif self.params.majority_vote == True:
                attacker_weight =  signWeight(self.updateReplace(model,weight,majority_vote=True))
                # attacker_weight = signWeight(subWeight(weight,self.updateReplace(model,weight)))
                # attacker_weight = signWeight(subWeight(self.updateReplace(model,weight),weight))
            elif self.params.rsa == True:
                print('rsa backdoor')
                # attacker_weight = signWeight(subWeight(self.updateReplace(model,weight),weight))
                attacker_weight = signWeight(subWeight(weight,self.updateReplace(model,weight)))
            else:
                attacker_weight =  self.updateReplace(model,weight)
            
        elif self.params.attack_type == 'pixel' and self.now_round % self.params.pixel_attack_frequency == 0:
            '''进行更改像素的后门攻击，生成攻击模型参数'''
            attacker_weight = self.updatePixel(model,weight)
        
        elif self.params.attack_type == 'label' and (self.now_round % self.params.labelflip_attack_frequency )== 0:
            attacker_weight = self.updateLabelflip(model,weight)
        
        elif self.params.attack_type == 'weightflip' and self.now_round % self.params.weightflip_attack_frequency == 0:
            attacker_weight = self.updateWeightflip(model,weight)
        elif self.params.attack_type == 'gaussian':
            attacker_weight = self.updateGaussaion(weight)
        else:
            if self.params.attack_type not in ['pixel','replace','label','weightflip']:
                print('你所输入的攻击不在攻击类型中')
            else:
                '''说明这一轮没有攻击'''
                attacker_weight = super().update(weight)

        return attacker_weight
    

    '''标签反转的后门攻击方式'''
    def updateLabelflip(self,model,weight):
        #用来留着进行替换，换成正常的数据
        original_dataloader = copy.deepcopy(self.dataloader)

        #进行投毒，相当于将dataloader换成malicious_dataloader 
        self.dataloader = self.malicious_dataloader

        print('Begin Attack: Backdoor Label Flip')
        #进行模型更新，已经增加过round不用再增加
        new_weight = super().update(weight)    
        model.load_state_dict(new_weight)
        #换回正常数据
        self.dataloader = copy.deepcopy(original_dataloader)

        #打印攻击模型的准确率损失
        self.backdoor_acc,self.backdoor_loss = self.inference(model,self.malicious_dataloader)
        self.printBackdoorInfo()   
        
        print('End Attack')

        return new_weight

    '''随机高斯噪声'''
    def updateGaussaion(self,weight):
        print('进行Gaussian Noise攻击')
        gaussain = {}
        for key,value in weight.items():
            mean = torch.mean(value)
            var = torch.var(value)
            gaussain[key] = torch.normal(mean,var,size=value.shape,device=value.device)
            if self.params.update_type=='direction':
                # gaussain[key] = torch.sign(torch.normal(mean,var,size=value.shape,device=value.device))
                gaussain[key] = torch.sign(torch.normal(mean,var,size=value.shape,device=value.device)-value)
        return gaussain

    def updateWeightflip(self,model,weight):

        print('Begin Attack: Weight Flip')
        if self.params.onud == True:
            new_weight = flipWeight(super().updateONUD(model,weight))
        else:
            new_weight = flipWeight(super().update(weight))
        print('End Attack')
        return new_weight



    '''在图片上添加像素的后门攻击方式'''
    def updatePixel(self,model,weight):

        #用来留着进行替换，换成正常的数据
        original_dataloader = copy.deepcopy(self.dataloader)

        #进行投毒，相当于将dataloader换成malicious_dataloader 
        self.dataloader = self.malicious_dataloader

        print('Begin Attack: Backdoor Pixel')
        #进行模型更新，已经增加过round不用再增加
        new_weight = super().update(weight)
       
        #换回正常数据
        self.dataloader = copy.deepcopy(original_dataloader)
        model.load_state_dict(new_weight)

        #打印攻击模型的准确率损失
        self.backdoor_acc,self.backdoor_loss = self.inference(model,self.malicious_dataloader)
        self.printBackdoorInfo()

        print('End Attack')

        return new_weight

    '''模型替换的后门攻击更新方式'''
    def updateReplace(self,model,weight,majority_vote=False):
        
        server_weight = copy.deepcopy(weight)
        # old_weight = copy.deepcopy(self.model.state_dict())

        # model = self.model
        model.load_state_dict(weight)

        model.train()

        device = self.device

        criterion = self.criterion

        optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr, momentum=0.5)

        # Client Epoch  
        self.client_epoch = 0

        print('Begin Attack: Backdoor Replace')
        alpha = 0.7
        for iter in range(15):
            for batch_idx,benign_data in enumerate(self.dataloader):
                mix_data,mix_target = dataReplace(self.params,benigns=benign_data,malicious=self.malicious_dataloader)
                mix_data,mix_target = mix_data.to(device),mix_target.to(device)
                optimizer.zero_grad()
                output = model(mix_data)
                class_loss = criterion(output,mix_target)
                # ano_loss = l2Weight(subWeight(model.state_dict(),server_weight))
                ano_loss = torch.norm(weightToVec(model.state_dict())-weightToVec(server_weight),p=2)
                # distance_loss = 
                loss = alpha*class_loss + (1-alpha)*ano_loss
                loss.backward()
                optimizer.step()
                if loss.item()<2.5:
                    break

            self.client_epoch = self.client_epoch + 1

        # 获取训练结果，打印正常人物信息
        self.train_acc,self.train_loss = self.inference(model,self.dataloader)
        self.print_acc_loss()
        
        # 打印后门信息
        self.backdoor_acc,self.backdoor_loss = self.inference(model,self.malicious_dataloader)
        self.printBackdoorInfo()
        
        self.model = model

        attack_model_weight = {}

        if majority_vote == True:
            for key, value in model.state_dict().items():
                attack_model_weight[key] =  -1 * (value - server_weight[key]) 
                # attack_model_weight[key] =  1 * (value - server_weight[key]) + server_weight[key]
                # attack_model_weight[key] =  -2000 * (value)
        # elif self.params.sign==True:
        #     for key, value in model.state_dict().items():
        #         attack_model_weight[key] = 100 * ( value - server_weight[key]) + server_weight[key]
    
        else:
            # L = 100*X-99*G = G + (100*X- 100*G) =  n / eta ( X - G) + G
            n = self.params.num_client
            m = self.params.attack_num
            eta = self.params.server_lr
            # S = 2
            # gamma = S/torch.norm(weightToVec(model.state_dict())-weightToVec(server_weight),p=2)
            gamma = 1
            for key, value in model.state_dict().items():
                # gamma = S/torch.norm((value-server_weight[key]),p=2)
                # print(gamma)
                # gamma = 10
                attack_model_weight[key] = gamma * ( value - server_weight[key]) + server_weight[key]
                # attack_model_weight[key] = gamma * ( value - server_weight[key])
                # attack_model_weight[key] = 20*value - 19*server_weight[key]
                # attack_model_weight[key] = value
                # attack_model_weight[key] = (n*value-(m-n)*server_weight[key])/m
        
        
        print('End Attack')

        self.now_round = self.now_round + 1 #恶意模型更新也要进行加一

        return attack_model_weight
   
    def printBackdoorInfo(self):
        print('Attack Type: {} \t Backdoor Acc: {:.5f}\t Backdoor Loss: {:.5f}'.format(self.params.attack_type,self.backdoor_acc,self.backdoor_loss))   



