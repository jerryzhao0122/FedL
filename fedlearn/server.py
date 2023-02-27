import imp
import math
import sys
from http import client
from pydoc import replace

import numpy as np
# from aiohttp import client
# from sklearn import cluster
import pandas as pd
# from soupsieve import select
from fedlearn.findAttacker import KMeans, findAttacker
from numpy.random.mtrand import normal
from privacy.privacy_tools import getGaussNoise
from sklearn import metrics

sys.path.append('..')
import copy

import torch
from opacus.accountants.utils import get_noise_multiplier
from tensorboardX import SummaryWriter
from torch import feature_alpha_dropout, nn
from utils.compute_weights import average_weights

from .attack.labelflip import getLabelFlip, poisonLabelFlip
from .attack.pixel import poisonPixeled
from .attack.replace import getBackdoorDataloader
from .helper import (acc_loss_inference, addWeight, boundWeight,
                     boundWeightAll, clusterSumSign, countSign, distWeight,
                     numMulWeight, rub1sumWeight, signToImg, signToImg2,
                     signWeight, subWeight, sumWeight, weightToVec, zeroWeight,my_NMI)


class Server(object):
    def __init__(self,model,test_dataloader,params,root_dataloader=None):
        self.params = params
        self.model = model
        self.round = int(0)
        self.root_dataloader = root_dataloader
        self.test_dataloader = test_dataloader
        # self./mnt/logs = /mnt/logs
        self.criterion = nn.CrossEntropyLoss()
        self.bengin_id = []
        self.malicious_id = []
        self.distM = []
        self.weight_vec_list = []
        self.mipc_results = {'M':[],'P':[],'MC':[],'HC':[]}

        fedl_config = 'client_num:' + str(params.num_client) + 'train_num:' + str(params.train_num) + 'train_round:' + str(params.round) + '(model):' + str(params.model) + '(experiment):' + str(params.name)
        if params.attack == True:
            if params.attack_type == 'replace':
                attack_config = 'attack_round:'+str(params.attack_replace_round)
            elif params.attack_type == 'pixel':
                attack_config = 'attack_num:'+str(params.attack_num)+'attack_frequency:'+str(params.pixel_attack_frequency)
            elif params.attack_type == 'label':
                attack_config = 'attack_num:'+str(params.attack_num)+'attack_frequency:'+str(params.labelflip_attack_frequency)
            elif params.attack_type == 'weightflip':
                attack_config = 'attack_num:'+str(params.attack_num)+'attack_frequency:'+str(params.weightflip_attack_frequency)
            elif params.attack_type == 'gaussian':
                attack_config = 'attack_num:'+str(params.attack_num)+'attack_frequency:'+str(params.gaussian_attack_frequency)
            else:
                print('攻击参数输入有误')
            
            # if params.rsa == True:
            #     result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'rsa' + '/attacked/' + str(params.attack_type) + '/' + fedl_config + attack_config
            # elif params.onud == True:
            #     if params.onud_root_data == True:
            #         result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'onud' + '/attacked/' + 'rootdata' + '/'  + str(params.attack_type) + '/' + fedl_config + attack_config
            #     else:
            #         result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'onud' + '/attacked/' + 'norootdata' + '/' + str(params.attack_type) + '/' + fedl_config + attack_config
            #     result_path = result_path + '/use_find:' + str(params.use_find)
            # else:
            result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'attack' + '/' + str(params.attack_type) + '/' + attack_config + fedl_config
        
        
        else:
            if params.rsa == True:
                if params.update_type == 'dirction':
                    result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'rsa' + '/' + 'dirction' + '/' + fedl_config
                else: 
                    result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'rsa' + '/' + 'normal' + '/' + fedl_config
            elif params.onud == True:
                if params.onud_root_data == True:
                    result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'onud' + '/' + 'normal' + '/rootdata/' + fedl_config
                else:
                    result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'onud' + '/' + 'normal' + '/norootdata/' + fedl_config
                result_path = result_path + '/use_find:' + str(params.use_find)
            else:
                result_path = '/home/featurize/result/tensorboard_logs/' + str(params.dataset)+ '/' + str(params.data_distributed)+ '/' + 'normal' + '/' + fedl_config
        print(result_path)
        self.writer = SummaryWriter(result_path)
    
    # def saveClientModel(self,values_dict):
    #     for key,values in values_dict.items():
    #         save_dir = ''

    def aggregation(self,values_dict):

        # if self.round in [1,100,150,200,250,300,500,1000,2000,3000,4000] and self.params.white_box:
        if self.round in [1,10,100,200,300,500,1000,2000,3000,4000] and self.params.white_box:
            print('save clients model')
            self.saveClientModel(values_dict)

        if self.params.robust_agg == True:
            print('鲁棒性聚合')
            if self.params.robust_agg_type == 'krum':
                updates = self.m_krum(values_dict)
            elif self.params.robust_agg_type == 'trummean':
                updates = self.trummean(values_dict)
            elif self.params.robust_agg_type == 'trummedian':
                updates = self.trummedian(values_dict)
            elif self.params.robust_agg_type == 'mipc':
                updates = self.mipc(values_dict,self.params.ad_dist)
            else:
                print('请输入正确的鲁棒聚合方法')
        else:
            updates = values_dict

        # if self.params.white_box == True:
        #     saveClientModel(values_dict)

        # self.model.cuda()
        print('开始聚合，当前轮数：',self.round)
        print('是否发动白盒攻击：',self.params.white_box)
        if self.params.rsa == True:
            self.aggRSA(updates)
        elif self.params.onud == True:
            print('USE ONUD')
            self.aggONUD(updates)
        elif self.params.sign == True or self.params.sign_v2==True:
            self.aggSIGN(updates)
        elif self.params.cdp == True:
            self.run_CDP(updates)
        elif self.params.majority_vote == True:
            self.aggMajorityVote(updates)
        else:
            print('Fed AVG')
            self.aggWeight(updates)

        # # 添加CDP
        # if self.params.cdp == True:
        #     self.run_CDP(updates)
            

        # if self.round in [1,100,150,200,250,300,1000,2000,3000,4000] and self.params.white_box:
        #     print('save server model')
        #     self.saveServerModel()
        # # self.model.cpu()


    def mipc(self,values_dict,dd='mi'):
        print('mipc_',dd)

        # 存在的参数
        N = self.params.train_num
        M = self.params.attack_num
        V = []
        Rho = M*(M-1)/(N*(N-1))

        # 保存sign图像用于分析
        for id, value in values_dict.items():
            # # 打印sign图像
            # if self.round == 1 or self.round%20==0:
            #     signToImg2(value,id,self.round,self.params)
            V.append(weightToVec(value))

        # 计算距离矩阵和截断距离
        M = np.random.rand(N,N)
        K = []
        for i in range(N):
            for j in range(i,N):
                if dd == 'mi':
                    # M[i][j] = 1 - metrics.normalized_mutual_info_score(V[i],V[j])
                    # import pdb;pdb.set_trace()
                    M[i][j] = 1 - my_NMI(V[i],V[j])
                elif dd == 'l2':
                    # M[i][j] = torch.norm(torch.cat([V[i],V[j]]))
                    M[i][j] = torch.norm(V[i]-V[j],p=2)
                else:
                    print('error dd:{}'.format(dd))

                if i!=j:
                    K.append(M[i][j])
                    M[j][i]=M[i][j]
        
        K_s = np.argsort(K)[math.ceil(len(K)*Rho)]
        d_c = K[K_s]

        # 计算密度
        P = []
        for i in range(N):
            pi = 0
            for j in range(N):
                if M[i][j]<d_c and i!=j:
                    pi = pi + 1
            P.append(pi)
        
        print('Densty',P)

        # 计算密度均值，小于均值的为好client
        P = np.array(P)
        P_mean = np.mean(P)
        normal_client = np.argwhere(P<=P_mean).reshape(-1).tolist()
        attack_client = np.argwhere(P>P_mean).reshape(-1).tolist()

        # 打印区别好的客户端和恶意客户端后的结果
        print('normal client: ',normal_client)
        print('attack_client: ',attack_client)

        if self.params.save_mipc == True:
            print('Save Mipc results')
            self.mipc_results['M'].append(M)
            self.mipc_results['P'].append(P)
            self.mipc_results['MC'].append(attack_client)
            self.mipc_results['HC'].append(attack_client)


        # 构建好的客户端组
        tmp_values_dict = {}
        for id, value in values_dict.items():
            if id in normal_client:
                tmp_values_dict[id]=values_dict[id]
        # print(tmp_values_dict)
        return tmp_values_dict

    def m_krum(self,values_dict):
        print('Krum')

        n = self.params.train_num
        f = self.params.attack_num
        m = n - f

        # 获取客户端的权重
        clients_weight = values_dict.values()

        num_d = n-f-1


        # 保存Kr分
        Kr=[]

        # 计算排序后的距离矩阵
        for i_weight in clients_weight:
            dist_i = []
            for j_weight in clients_weight:
                dist_ij  = distWeight(i_weight,j_weight)
                dist_i.append(dist_ij)
            dist_i = sorted(dist_i)
            Kr_i = sum(dist_i[1:num_d+1])
            Kr.append(Kr_i.cpu().item())
        
        # 计算排序索引
        # print(Kr)
        np_Kr = np.array(Kr)
        Kr_argsort = list(np_Kr.argsort())
        # print(Kr_argsort)
        bengin_idx = Kr_argsort[:m]
        macilous_idx = Kr_argsort[m:]

        tmp_values_dict = {}
        for i in bengin_idx:
            tmp_values_dict[i] = values_dict[i]
        print('bengin_client: ',bengin_idx)
        print('macilous_idx:',macilous_idx)
        return tmp_values_dict

    def trummean(self,values_dict):
        print('Trummean')
        k = int(self.params.attack_num/2)

        # 获取客户端的权重
        clients_weight = values_dict.values()

        tmp = {}
        for key,value in values_dict[0].items():
            all_values = [i[key] for i in clients_weight]
            updates = torch.stack(all_values)
            largest, _ = torch.topk(updates, k, 0)
            neg_smallest, _ = torch.topk(-updates, k, 0)
            new_stacked = torch.cat([updates, -largest, neg_smallest]).sum(0)
            new_stacked /= len(updates) - 2 * k
            tmp[key] = new_stacked
        
        tmp_values_dict = {}
        tmp_values_dict[0] = tmp
        return tmp_values_dict
        
    def trummedian(self,values_dict):
        print('trummedian')
        clients_weight = values_dict.values()
        
        tmp={}
        for key,value in values_dict[0].items():
            all_values = [i[key]  for i in clients_weight]
            tmp1,_ = torch.median(torch.stack(all_values),dim=0)
            tmp2,_ = torch.median(-1*torch.stack(all_values),dim=0)
            tmp[key] = (tmp1 + tmp2)/2
        
        # 为了与聚合方法形成一致
        tmp_values_dict = {}
        tmp_values_dict[0]=tmp
        return clients_weight

    def aggMajorityVote(self,direction_dict):
        print('agg use Majority Vote')
        all_direction = list(direction_dict.values())
        sign_sum_direction = signWeight(sumWeight(all_direction))

        old_weight = copy.deepcopy(self.model.state_dict())
        new_weight = subWeight(old_weight, numMulWeight(self.params.majority_vote_eta, addWeight(sign_sum_direction,numMulWeight(self.params.majority_vote_lambda,old_weight))))
        self.model.load_state_dict(new_weight) 


    def aggSIGN(self, direction_dict):
        print('agg use SIGN')
        all_direction = list(direction_dict.values())
        sum_direction = sumWeight(all_direction)

        if 'topic' in str(self.params.name):
            print('用于实验画图')
            # 保存sign图像用于分析
            for id, value in direction_dict.items():
                # 打印sign图像
                if self.round == 1 or self.round%20==0:
                    signToImg2(value,id,self.round,self.params)

        # 设定动态参数
        if self.round==1:
            self.alpha = self.params.sign_alpha 
        else:
            if self.params.white_box == True:
                if self.round % 50 == 0 and self.alpha>=0.00001:
                    self.alpha = self.alpha / 2
            elif self.params.model == 'alexnet':
                if (self.round % 300 == 0) and self.alpha>=0.00001:
                    self.alpha = self.alpha / 2
            else:
                if self.round % 100 == 0 and self.alpha>=0.00001:
                    self.alpha = self.alpha / 2
            
            if self.params.model == 'resnet':
                if self.round % 20 == 0 and self.alpha>=0.000001:
                    self.alpha = self.alpha / 2
        print('服务器学习率：',self.alpha)

        # 更新模型参数
        old_weight = copy.deepcopy(self.model.state_dict())
        

        new_weight = addWeight(old_weight,numMulWeight(self.alpha,sum_direction))
        self.model.load_state_dict(new_weight) 
        
        # print('sum_direction',sum_direction['fc3.1.bias'])
        # print('old',old_weight['fc3.1.bias'])
        # print('new',new_weight['fc3.1.bias'])
        # print('model',self.model.state_dict()['fc3.1.bias'])


    def aggONUD(self,values_dict):

        # 使用误差修正方法
        # dirction_and_error_list = list(values_dict.values())
        # dirction_list = [i[0] for i in dirction_and_error_list]
        # error_list = [i[1] for i in dirction_and_error_list]
        print('agg use ONUD')

        # dirction_list = list(values_dict.values())
        dirction_list = []
        
        for id, value in values_dict.items():
            # print(id)
            # 打印sign
            if self.round == 1 or self.round%10==0:
                signToImg2(value,id,self.round,self.params)
            # 获得所有的方向
            dirction_list.append(value)

        old_weight = copy.deepcopy(self.model.state_dict())
        # old_weight = self.model.state_dict()
        # self.sum_dirction = sumWeight(dirction_list)

        # 使用findAttack进行聚合
        if self.params.use_find == True:
            dirction_vec_list = [weightToVec(d) for d in dirction_list]
            # normal_id,attack_id = KMeans(dirction_vec_list)
            normal_id,attack_id,M = findAttacker(dirction_vec_list,self.params.cfdp_rho,self.params.cfdp_alpha,self.params.cfdp_beta)
            print('normal_id:{}\nattack_id:{}'.format(normal_id,attack_id))
            normal_dirction_list = [dirction_list[i] for i in normal_id]
            self.sum_dirction = sumWeight(normal_dirction_list)

            # 保存重要数据
            self.bengin_id.append(normal_id)
            self.malicious_id.append(attack_id)
            self.distM.append(M)
            self.weight_vec_list.append(list(torch.tensor(dirction_vec_list).cpu()))
        else:
            self.sum_dirction = sumWeight(dirction_list)

        # 使用-1，0，1的统计特征进行聚类求和
        # self.sum_dirction = clusterSumSign(dirction_list,feature=False)
        # self.sum_dirction = sumWeight(dirction_list)
        
        # self.sum_error = sumWeight(error_list) # 误差

        # 动态调整学习率，防止过拟合造成loss反弹
        if self.round==1:
            self.alpha = self.params.onud_server_alpha 
        else:
            if self.round % 10 == 0 and self.alpha>=0.00000001:
                self.alpha = self.alpha / 2

        # # 固定学习率
        # self.alpha = self.params.onud_server_alpha 

        # 计算rootdataset的梯度信息，用于修正参数
        if self.params.onud_root_data == True: #当使用rootdata进行修正时候
            model_grad = zeroWeight(self.sum_dirction)
            for iter,(data,traget) in enumerate(self.root_dataloader):
                # optimizer.zero_grad()
                data,traget = data.cuda(),traget.cuda()
                outputs = self.model(data)
                loss = self.criterion(outputs,traget)
                loss.backward()
                for p,(key,value) in zip(self.model.parameters(),self.sum_dirction.items()):
                    model_grad[key] = p.grad.data + model_grad[key]    
                # break
                # optimizer.step()
        print('服务器学习率：',self.alpha)
        
        # now_server_weight = self.model.state_dict()
        # self.broadcast_dirction = signWeight(subWeight(old_weight,now_server_weight))
        # 正常聚合
        # print(self.sum_dirction)
        server_weight = addWeight(old_weight,numMulWeight(self.alpha,self.sum_dirction))
        # 误差修正聚合
        # self.fix_dirction = addWeight(self.sum_dirction,numMulWeight(0.1,self.sum_error))
        # server_weight = subWeight(old_weight,numMulWeight(self.alpha,self.fix_dirction))
        # server_weight = subWeight(server_weight,numMulWeight(0.05*self.alpha,self.sum_error))

        if self.params.onud_root_data == True: #当使用rootdata进行修正时候
            server_weight = subWeight(server_weight,numMulWeight(2*self.alpha,model_grad))
            self.sum_dirction = addWeight(numMulWeight(2,signWeight(model_grad)),self.sum_dirction)
            # signToImg(signWeight(model_grad),'server',self.round)
        
        # 给定服务器模型的方向信息
        # self.sum_dirction = subWeight(self.sum_dirction,numMulWeight(10,signWeight(server_weight)))
       
        # server_weight = subWeight(old_weight,numMulWeight(self.params.onud_server_alpha,self.sum_dirction))
        self.model.load_state_dict(server_weight) 
    
    def aggRSA(self,values_dict):
        print('RSA_',self.params.update_type)
        values_list = list(values_dict.values())
        old_weight = copy.deepcopy(self.model.state_dict())
        if self.params.update_type == 'direction': # rsa只上传方向信息
            client_sign = sumWeight(values_list)
        else:
            client_sign = sumWeight([signWeight(subWeight(old_weight,i)) for i in values_list])
        new_weight = subWeight(old_weight, numMulWeight(self.params.rsa_alpha,(addWeight(numMulWeight(self.params.rsa_l1_lambda, client_sign), numMulWeight(self.params.rsa_weight_lambda, old_weight)))))
        self.model.load_state_dict(new_weight)

    def aggWeight(self,values_dict):
        weights_list = list(values_dict.values())
        aw = average_weights(weights_list)
        self.model.load_state_dict(aw)
    
    def broadcast(self):
             
        if self.params.sign == True:
            server_weight = self.broadcastSIGN()
        else:
            server_weight = self.model.state_dict()

        if self.round in [1,10,300,1000,2000,3000,4000] and self.params.white_box:
            print('save server model round ', self.round)
            self.saveServerModel(weight=server_weight)
        
        self.round = self.round + 1
        
        return server_weight
        # self.model.cpu()
    
    def broadcastSIGN(self):
        if self.params.sign_mask==True:
            masked_weight = self.run_mask()
            return masked_weight
        elif self.params.sign_noise==True:
            noised_weight = self.run_noise()
            return noised_weight
        else:
            return self.model.state_dict()

    def broadcastONUD(self):
        return self.model.state_dict()

        # if self.round == 1:
        #     # return signWeight(zeroWeight(self.model.state_dict()))
        #     return (signWeight(zeroWeight(self.model.state_dict())), signWeight(self.model.state_dict()),self.model.state_dict())
        #     # return signWeight(self.model.state_dict()),signWeight(self.model.state_dict())
        # else:
        #     # return self.broadcast_dirction
        #     # return signWeight(self.sum_dirction)
        #     return (signWeight(self.sum_dirction),signWeight(self.model.state_dict()),self.model.state_dict())
        #     # return (signWeight(self.sum_dirction),signWeight(self.sum_error))

    def run_mask(self):
        p = self.params.sign_mask_p
        weight = self.model.state_dict()
        masked_weight = {}
        for key,value in weight.items():
            random_mask = torch.rand_like(value,device=value.device)
            mask_matrix = torch.ones_like(value,device=value.device)
            mask_matrix[random_mask < p] = 0
            # print(mask_matrix)
            masked_weight[key] = value*mask_matrix

        return masked_weight

    def run_noise(self):
        weight = self.model.state_dict()
        noised_weight = {}
        for key,value in weight.items():
            noise = torch.normal(0,self.params.sign_noise_sigma,size=value.shape,device=value.device)
            noised_weight[key] = value + noise

        return noised_weight

    def run_CDP(self,value_dict):
        # 简单的只进行参数添加噪声
        server_weight = self.model.state_dict()
        S, bonded_weights = boundWeightAll(server_weight,value_dict)
        # S = self.params.cdp_clip

        sigma = get_noise_multiplier(
            target_delta=self.params.cdp_delta,
            target_epsilon=self.params.cdp_epsilon,
            sample_rate=1,
            steps=1
        )
        print('Server agg with CDP, epsilon: {} delta: {} sigma: {} S: {}'.format(self.params.cdp_epsilon,self.params.cdp_delta,sigma,S))

        # 对boundedweight求和
        sum_bounded_weight = sumWeight(bonded_weights)

        # print(server_weight)
        for key,value in server_weight.items():
            # print(value)
            g_noise = 0.1 * torch.normal(0,sigma*S,size=value.shape,device=value.device)
            server_weight[key] = server_weight[key] + (1/self.params.train_num)*(sum_bounded_weight[key]+g_noise)

        self.model.load_state_dict(server_weight)

    def writerLogs(self):
        self.writer.add_scalar('Server Loss',self.test_loss, self.round)
        self.writer.add_scalar('Server Acc', self.test_acc, self.round)
        if self.params.attack == True and self.params.attack_type != 'weightflip' and self.params.attack_type != 'gaussian':
            self.writer.add_scalar('Attack Loss',self.attack_loss,self.round)
            self.writer.add_scalar('Attack Acc',self.attack_acc,self.round)
        
    def printinfo(self):
        
        '''打印没有攻击情况下的数据'''
        self.test_acc, self.test_loss = acc_loss_inference(self.model,self.test_dataloader,self.criterion)
        print('Server Round: {}\t Test_Acc: {:.5f} \t Test_Loss: {:.5f}'.format(self.round,self.test_acc,self.test_loss))
        
        if self.params.attack == True:
            '''打印攻击的数据'''
            if self.params.attack_type == 'replace':
                replace_test_dataloader = getBackdoorDataloader(self.params)
                self.attack_acc, self.attack_loss = acc_loss_inference(self.model,replace_test_dataloader,self.criterion)
                # print('Attack type: {}\t Acc: {}\t Loss: {} \n'.format(self.params.attack_type,replace_acc,replace_loss))
            elif self.params.attack_type == 'pixel':
                pixeled_dataloader = poisonPixeled(self.params,self.test_dataloader)
                self.attack_acc, self.attack_loss = acc_loss_inference(self.model,pixeled_dataloader,self.criterion)
                # print('Attack type: {}\t Acc: {}\t Loss: {} \n'.format(self.params.attack_type,attack_acc,attack_loss))
            elif self.params.attack_type == 'label':
                labelflip_dataloader = getLabelFlip(self.params,self.test_dataloader)
                self.attack_acc, self.attack_loss = acc_loss_inference(self.model,labelflip_dataloader,self.criterion)
            else:
                pass
            
            if self.params.attack_type != 'weightflip' and self.params.attack_type!='gaussian':
                print('Attack type: {}\t Acc: {}\t Loss: {} \n'.format(self.params.attack_type,self.attack_acc,self.attack_loss))
            else:
                print('Rubost Attack')

    def saveInfo(self):
        info = pd.DataFrame()
        info['round'] = [i for i in range(1,self.round+1)]
        info['bengin_id'] = self.bengin_id
        info['malicious_id'] = self.malicious_id
        info['distM'] = self.distM
        vec_list = np.array(self.weight_vec_list)
        M = self.distM
        name = '/mnt/result/trainInfo/' + self.params.data_distributed + self.params.name[1:-1] +'.csv'
        name_2 = '/mnt/result/trainInfo/'+ self.params.data_distributed + self.params.name[1:-1] +'.npy' 
        name_3 ='/mnt/result/trainInfo/'+ self.params.data_distributed + self.params.name[1:-1] +'M.npy' 
        info.to_csv(name)
        np.save(name_2,vec_list)
        np.save(name_3,M)
    
    def saveServerModel(self,weight=None):
        path = '/home/featurize/result/models/server/' + str(self.params.data_distributed) + '_' + str(self.params.name) + '_' + str(self.round) + '.pth.tar'
        if weight == None:
            torch.save(self.model.state_dict(),path)
        else:
            torch.save(weight,path)
        print('server模型保存成功')
    
    def saveClientModel(self,values_dict):
        for key,value in values_dict.items():
            path = '/home/featurize/result/models/client/' + str(self.params.data_distributed) + '_' + str(self.params.name) + '_' +'client' +str(key) +'_round' + str(self.round) + '.pth.tar'
            torch.save(value,path)
        
        print('clients模型保存成功')
