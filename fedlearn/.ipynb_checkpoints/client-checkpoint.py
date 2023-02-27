'''
time: 2022/1/24
author: Guo Zhenyuan
'''
from hmac import new
import imp
import sys

# from grpc import server

from fedlearn import attack
from privacy.privacy_tools import getDpLr
sys.path.append('..')

from dataclasses import replace
from operator import mod
from statistics import mode
import torch
import copy
import numpy as np
# import yaml
# import gc

from torch import ne, nn

from tensorboardX import SummaryWriter

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from .helper import bondsignWeight, flipWeight, mulWeight, sameDircWeight, sortDict, subWeight, addWeight,numMulWeight,signWeight, sumWeight,zeroWeight,stosignWeight,l2Weight
from .attack.replace import getBackdoorDataloader,dataReplace
from .attack.pixel import poisonPixeled
from .attack.labelflip import poisonLabelFlip,getLabelFlip

class Clients():
    def __init__(self,dataloaders_list,params):
        self.params = params
        self.all_clients={}
        for id,dataloader in enumerate(dataloaders_list):
            client = Client(id,dataloader,1,params) # 开始时候round=1
            self.all_clients[id]=client
    
    def update(self,model,weight):
        '''
        return 一个字典{客户机id: 模型参数}，包含了当前round中选中的客户机
        '''
        num_update_client = max(int(self.params.train_num),1)
        ids_update_client = np.random.choice(range(self.params.num_client),num_update_client,replace=False)
        update_weights = {}
        for id in ids_update_client:
            update_client = self.all_clients[id]
            new_weight = update_client.update(model,copy.deepcopy(weight))
            # new_weight = update_client.update(weight)
            update_weights[id] = new_weight
        return sortDict(update_weights)
        

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

    def update(self,model,weight):
        # self.model.cuda()
        if self.params.update_type == 'weight':
            print('weight')
            new_weight = self.updateWeight(model,weight)
        elif self.params.update_type == 'dirction' and self.params.onud == True:
            print('you only need dirction')
            new_weight = self.updateONUD(weight)
        elif self.params.update_type == 'dirction' and self.params.rsa == True:
            print('rsa, client update dirction')
            new_weight = self.updateWeight(weight)
        else:
            print('请输入正确的更新类型')
        self.now_round = self.now_round + 1
        # self.model.cpu()
        torch.cuda.empty_cache()
        return new_weight
    
    def updateONUD(self,model,dirct):
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
        self.lam = self.params.onud_client_lambda
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

        model.load_state_dict(new_weight)
        
        # 能够获取模型细节信息
        model = extend(model)
        model.train()

        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        # 优化器
        # optimizer = self.optimizer
        optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr,momentum=0.5)

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
        dirction = bondsignWeight(subWeight(tmp_weight,now_weight),0.00000005)
        # error = signWeight(subWeight(subWeight(new_weight,now_weight),dirction))
        # dirction = stosignWeight(dirct_weight)
        # print(dirction)
        # torch.cuda.empty_cache()

        # dirction_and_error = [dirction,error]
        return dirction
        # return (dirction,error)


    def updateWeight(self,model,weight):
        '''
        返回的是整个模型的情况
        '''
        # old_weight = copy.deepcopy(self.model.state_dict())
        # model = self.model
        model.load_state_dict(weight)

        # server的参数
        server_weight = copy.deepcopy(weight)

        # 能够获取模型细节信息
        model = extend(model)
        
        model.train()
        
        # 选择训练位置
        device = self.device

        # 目标函数选择
        criterion = self.criterion

        # 设置优化器
        
        if self.params.rsa == True:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.alpha,momentum=0.5)
        else:
            optimizer = torch.optim.SGD(model.parameters(),lr = self.params.client_lr, momentum=0.5)

        # Client Epoch  
        self.client_epoch = 0

        for iter in range(self.params.client_ep):
            batch_loss = []
            for batch_idx,(data,target) in enumerate(self.dataloader):
                data,target = data.to(device),target.to(device)
                # print(data.shape)
                optimizer.zero_grad()
                log_probs = model(data)
                loss = criterion(log_probs,target)
                loss.backward()

                if self.params.ldp == True: # 使用了dp-sgd
                    pass
                elif self.params.rsa == True: # 使用了rsa提高鲁棒性
                    for p,sw in zip(model.parameters(),server_weight.values()):
                        p.grad.data = p.grad.data + self.params.l1_lambda * torch.sign(sw-p)
                        # p.grad.data = p.grad.data
        
                else:
                    # log_probs = model(data)
                    # loss = criterion(log_probs,target)
                    # loss.backward()
                    pass
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
    def __init__(self, id, dataloader, model, now_round, params):
        super(Attacker,self).__init__(id, dataloader, model, now_round ,params)
        if params.attack_type == 'replace':
            self.malicious_dataloader = getBackdoorDataloader(self.params)
        elif params.attack_type == 'pixel':
            self.malicious_dataloader = poisonPixeled(self.params,self.dataloader)
        elif params.attack_type == 'label':
            # self.malicious_dataloader = poisonLabelFlip(self.params,self.dataloader)
            self.malicious_dataloader = getLabelFlip(self.params,self.dataloader)
        else:
            pass


    def update(self,model, weight):
        
        # 如果是onud时候获取的是方向信息，因此要变换weight
        # if self.params.onud == True and self.params.attack_type == 'replace':
            # old_weight = copy.deepcopy(self.model.state_dict())
            # dirction = copy.deepcopy(weight)
            # weight = copy.deepcopy(subWeight(old_weight,numMulWeight(self.params.onud_client_lambda,dirction)))

        if self.params.attack_type=='replace' and self.params.attack_replace_round <= self.now_round:
            '''进行模型替换的后门攻击，生成攻击的模型参数'''
            if self.params.onud == True:
                attacker_weight =  signWeight(subWeight(weight[2],self.updateReplace(model,weight[2])))
            else:
                attacker_weight =  self.updateReplace(model,weight)
            
        elif self.params.attack_type == 'pixel' and self.now_round % self.params.pixel_attack_frequency == 0:
            '''进行更改像素的后门攻击，生成攻击模型参数'''
            attacker_weight = self.updatePixel(model,weight)
        
        elif self.params.attack_type == 'label' and self.now_round % self.params.labelflip_attack_frequency == 0:
            attacker_weight = self.updateLabelflip(model,weight)
        
        elif self.params.attack_type == 'weightflip' and self.now_round % self.params.weightflip_attack_frequency == 0:
            attacker_weight = self.updateWeightflip(model,weight)
        
        else:
            if self.params.attack_type not in ['pixel','replace','label','weightflip']:
                print('你所输入的攻击不在攻击类型中')
            else:
                '''说明这一轮没有攻击'''
                attacker_weight = super().update(model,weight)

        return attacker_weight
    

    '''标签反转的后门攻击方式'''
    def updateLabelflip(self,model,weight):
        #用来留着进行替换，换成正常的数据
        original_dataloader = copy.deepcopy(self.dataloader)

        #进行投毒，相当于将dataloader换成malicious_dataloader 
        self.dataloader = self.malicious_dataloader

        print('Begin Attack: Backdoor Label Flip')
        #进行模型更新，已经增加过round不用再增加
        new_weight = super().update(model,weight)    
 
        #换回正常数据
        self.dataloader = copy.deepcopy(original_dataloader)

        #打印攻击模型的准确率损失
        self.backdoor_acc,self.backdoor_loss = self.inference(self.model,self.malicious_dataloader)
        self.printBackdoorInfo()   
        
        print('End Attack')

        return new_weight

    def updateWeightflip(self,model,weight):

        print('Begin Attack: Weight Flip')
        if self.params.onud == True:
            new_weight = flipWeight(super().updateONUD(model,weight))
        else:
            new_weight = flipWeight(super().update(model,weight))
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
        new_weight = super().update(model,weight)
       
        #换回正常数据
        self.dataloader = copy.deepcopy(original_dataloader)

        #打印攻击模型的准确率损失
        self.backdoor_acc,self.backdoor_loss = self.inference(self.model,self.malicious_dataloader)
        self.printBackdoorInfo()

        print('End Attack')

        return new_weight


    '''模型替换的后门攻击更新方式'''
    def updateReplace(self,model,weight):
        
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
        for iter in range(10):
            for batch_idx,benign_data in enumerate(self.dataloader):
                mix_data,mix_target = dataReplace(self.params,benigns=benign_data,malicious=self.malicious_dataloader)
                mix_data,mix_target = mix_data.to(device),mix_target.to(device)
                optimizer.zero_grad()
                output = model(mix_data)
                class_loss = criterion(output,mix_target)
                # distance_loss = 
                loss = class_loss
                loss.backward()
                optimizer.step()

            self.client_epoch = self.client_epoch + 1

            # 获取训练结果，打印正常人物信息
            self.train_acc,self.train_loss = self.inference(model,self.dataloader)
            self.print_acc_loss()
        
        # 打印后门信息
        self.backdoor_acc,self.backdoor_loss = self.inference(model,self.malicious_dataloader)
        self.printBackdoorInfo()
        
        self.model = model

        attack_model_weight = {}

        # L = 100*X-99*G = G + (100*X- 100*G) =  n / eta ( X - G) + G
        n = self.params.num_client
        eta = self.params.server_lr
        for key, value in model.state_dict().items():
            # attack_model_weight[key] = n/eta * ( value - server_weight[key]) + server_weight[key]
            attack_model_weight[key] = 20*value - 19*server_weight[key]
        
        print('End Attack')

        self.now_round = self.now_round + 1 #恶意模型更新也要进行加一

        return attack_model_weight
   
    def printBackdoorInfo(self):
        print('Attack Type: {} \t Backdoor Acc: {:.5f}\t Backdoor Loss: {:.5f}'.format(self.params.attack_type,self.backdoor_acc,self.backdoor_loss))   



