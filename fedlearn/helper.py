'''
time:2022/2/25
author: GuoZhenyuan
'''

from pandas import value_counts
from rsa import sign
import torch
import math
import numpy as np
import torch
import copy
import cv2
import os
from collections import OrderedDict
from collections import Counter
import numpy as np 
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster
import scipy.cluster.hierarchy as sch


def bound(xi_list,bound_type = 'median'):
    #计算边界值S

    xis = xi_list

    #整理一个包含所有xi的字典列表{lay1:[tenosr1,tensor2......]}
    S_all={}
    #将所有的xi放到字典列表中，便于计算均值中位数等等
    for xi in xis:
        for key, value in xi.items():
            if key not in S_all.keys():
                S_all[key] = [value]
            else:
                S_all[key].append(value)
    #用于记录每一层的边界值
    if bound_type == 'mean':
        pass
        # S=copy.deepcopy(xis[0])
        # for xi in xis[1:]:
        #     for key,value in xi:
        #         S[key] = S[key] + value
    elif bound_type == 'median':
        # 返回的是一个{lay1:lay1_S,lay2:lay2_S}
        S_median = {}
        for key,value in S_all.items():
            S_median[key] = np.median(value)
        return S_median

def clipAll(weight):
    weight_vector = torch.FloatTensor([])
    for key, value in weight.items():
        weight_vector = torch.cat(weight_vector,value.view(-1))
    
    return weight_vector



def clipLayer(weight,xi_list,bound):
    #对每一层数据进行一次裁切，如果大于相当于w/max(1,xi/S)
    w = weight
    S = bound
    xis = xi_list

    # 返回值所有裁切后的local的state_dict_list
    local_clipped_weight = []
    # clipped_w = {}
    # 遍历所有客户机的权重和对应的l2范数xi
    for local_num in range(len(w)):

        # 获得某个客户机的state_dict，和l2范数dict
        l_w = w[local_num]
        l_xi = xis[local_num]
        
        # 单个客户机裁切后的值
        lcw = {}
        for key, value in l_w.items():
            l_xi_layer = l_xi[key]
            l_w_layer = value
            # 计算每一层被裁切过后的值，相当于△w/max(1, ζ/S）
            l_w_layer_clipped = l_w_layer / (max(1.0,float(l_xi_layer/S[key])))
            
            lcw[key] = l_w_layer_clipped

            # if key not in clipped_w.keys():
            #     clipped_w[key] = [l_w_layer_clipped]
            # else:
            #     clipped_w[key].append(l_w_layer_clipped)

        # 整合所有的客户机裁切后的值
        local_clipped_weight.append(lcw)

    return local_clipped_weight


def gaussNoise(weight,S,sigma):
    
    gauss_dic = {}
    # 遍历每一层

    for key, value in weight.items():
        #print(S[key])
        lay_noise = torch.tensor(np.random.normal(0,float(S[key]*sigma),value.shape))
        gauss_dic[key] = lay_noise
    
    return gauss_dic


# weight 计算操作
def addWeight(model_dict1,model_dict2):
    new_model_dict={}
    for key, value in model_dict1.items():
        new_model_dict[key] = model_dict2[key] + value
    
    return new_model_dict

def subWeight(model_dict1,model_dict2):
    '''weight1 - weight2'''
    new_model_dict={}
    for key,value in model_dict2.items():
        new_model_dict[key] = model_dict1[key] - value
    return new_model_dict

def sumWeight(model_dict_list):
    '''sum(weight_list)'''
    new_model_dict = {}
    for key, value in model_dict_list[0].items():
        new_model_dict[key] = sum([model_dict[key] for model_dict in model_dict_list])
    return new_model_dict

def mulWeight(model_dict1,model_dict2):
    tmp = {}
    for key, value in model_dict1.items():
        tmp[key] = value * model_dict2[key]
    return tmp

def rub1sumWeight(model_dict_list):
    new_model_dict = {}
    num_dict = len(model_dict_list)
    start_idx = math.ceil(num_dict/0.25)
    end_idx = math.floor(num_dict/0.75)
    for key, value in model_dict_list[0].items():
        all_value = torch.stack(tuple([model_dict[key] for model_dict in model_dict_list]))
        all_value_sort,_ = torch.sort(all_value,0) # dim=0 对其值按照0维进行排序
        new_model_dict[key] = torch.sum(all_value_sort[start_idx:end_idx])
    return new_model_dict

def numMulWeight(number,weight):
    # num x weight
    new_model_dict = {}
    for key, value in weight.items():
        new_model_dict[key] = number * value
    return new_model_dict

def signWeight(weight):
    tmp = {}
    for key,value in weight.items():
        tmp[key] = torch.sign(value)
    return tmp

def bondsignWeight(weight,bond):
    tmp = {}
    for key,value in weight.items():
        if 'running' in key:
            tmp[key] = value
        else:
            value_abs = torch.abs(value)
            tmp_value = torch.where(value_abs<bond,torch.full_like(value,0),value)
            tmp[key] = torch.sign(tmp_value)
    return tmp

def stosignWeight(weight):
    tmp={}
    b = 0
    for value in weight.values():
        max_tmp = torch.max(torch.abs(value))
        if b < max_tmp :
            b = max_tmp 
    
    for key,value in weight.items():
        r = torch.rand(value.shape)
        tmp_pro = (value + b) / 2 * b
        tmp_a = torch.where(tmp_pro<r,torch.full_like(tmp_pro,1),torch.full_like(tmp_pro,-1))
        tmp_c = torch.where(tmp_pro == 1.0/2,torch.full_like(tmp_pro,0),tmp_a) 
        tmp[key]=tmp_c
    return tmp

def l2Weight(weight):
    l2 = 0
    for value in weight.values():
        # print('ok')
        # print(len(value.shape))
        if len(value.shape)>0:
            tmp = torch.norm(value,p=2)
            l2 = l2 + tmp
    return l2
    
def zeroWeight(weight):
    tmp={}
    for key,value in weight.items():
        tmp[key] = torch.zeros_like(value)
    return tmp

def flipWeight(weight):
    tmp={}
    for key,value in weight.items():
        tmp[key] = -1 * value
    return tmp

def weightToVec(weight):
    tmp = []
    for key,value in weight.items():
        tmp.append(value.view(-1))
    return torch.cat(tmp)

def distWeight(weight_a,weight_b):
    dist = 0
    for key,value in weight_a.items():
        dist = dist + torch.dist(weight_a[key],weight_b[key],p=2)
    return dist


# -------



def mulScalarDict(scalar,dic):
    new_dict={}
    for key, value in dic.items():
        new_dict[key] = value * scalar
    return new_dict

def dp_noise(param, S=1, sigma=0.5):
    # 输入参数矩阵 输出噪声矩阵
    noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=S*sigma)
    
    return noised_layer

def weight_add_dp_noise(weight,S,sigma):
    
    for key,value in weight.items():
        # print(S[key])
        weight[key] = weight[key] + dp_noise(value,S[key],sigma)
    
    return weight



def get_mask_matrix(matrix,mask_r=0.1):
    '''返回矩阵的mask矩阵'''
    sp = matrix.shape
    n = np.prod(sp)
    r = mask_r
    zeros = torch.zeros(int(n*r))
    ones = torch.ones(n-int(n*r))
    mask_list = torch.cat([zeros,ones])
    mask_matrix = mask_list[torch.randperm(mask_list.size(0))].reshape(sp)
    return mask_matrix


def maskWeights(weight):
    '''对参数进行mask，返回mask后的参数和mask_matrix'''
    mask = {}
    masked_value = {}
    for key, value in weight.items():
        value_mask_matrix = get_mask_matrix(value)
        mask[key] = value_mask_matrix
        # print(value.shape)
        # print(value_mask_matrix.shape)
        masked_value[key] = value * value_mask_matrix
    
    # return masked_value
    return masked_value,mask


def acc_loss_inference(model,dataloader,criterion):
    model.eval()
    loss,acc,total = 0.0,0.0,0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, (data,target) in enumerate(dataloader):
        data,target = data.to(device), target.to(device)
        # print(data.shpae)
        outputs = model(data)
        batch_loss = criterion(outputs,target)
        loss = loss + float(batch_loss.data)
        
        _,preds = torch.max(outputs,1)
        preds = preds.view(-1)
        acc = acc + torch.sum(torch.eq(preds,target)).item()
        total = total + len(target)

    acc = acc / total
    # loss = loss / total
    return acc, loss
        

def signToImg(sign,clientid,round):
    '''用于将上传的sign转换为图片 用于分析下为什么不会聚合'''
    for key,value in sign.items():
        tmp = str(key).replace('.','-')
        value_shape = value.shape
        # print(len(value_shape))
        if len(value_shape)<2:
            h = value_shape[-1]
        else:
            h = value_shape[-2]

        kk = value.view(-1)
        tt = torch.where(kk==0,torch.full_like(kk,125),kk)
        tt = torch.where(kk==1,torch.full_like(kk,255),tt)
        tt = torch.where(kk==-1,torch.full_like(kk,0),tt)
        tt = tt.view((h,-1))
        
        img = tt.cpu().numpy()
        # print(img)

        dir = './result/' + 'client:'+ str(clientid) + '/' + 'round:' + str(round) + '/' 
        if os.path.exists(dir) == False:
            os.makedirs(dir) 
        name = dir + str(tmp) + '.jpg'
        
        cv2.imwrite(name,img)
        # print('保存signtoimg',name)

def signToImg2(sign,clientid,round,params):
    '''用于将上传的sign转换为图片 用于分析下为什么不会聚合'''
    tmp = []
    for key,value in sign.items():
        kk = value.view(-1)
        tmp.append(kk)
    tmp = torch.cat(tmp)
    tt = torch.where(tmp==0,torch.full_like(tmp,125),tmp)
    tt = torch.where(tmp==1,torch.full_like(tmp,255),tt)
    tt = torch.where(tmp==-1,torch.full_like(tmp,0),tt)
    l = len(tt)
    w = math.ceil(l**0.5)
    f = w-l%w
    tt = torch.cat([tt,torch.zeros(f).cuda()])

    img = tt.view(w,-1).cpu().numpy()

    dir = '/home/featurize/result/signimg/' + params.name +'/' + params.data_distributed  + '/' + str(round) + '/'
    if os.path.exists(dir) == False:
        os.makedirs(dir) 
    name = dir + 'client:'+ str(clientid) + '.png'
        
    cv2.imwrite(name,img)
    # print('保存signtoimg',name)


def countSign(sign):
    tmp = {}
    tmp_count = {-1:0,0:0,1:0}
    for key, value in sign.items():
        tmp_value = value.view(-1).cpu().numpy()
        count = Counter(tmp_value)
        tmp_count[-1] = count[-1] + tmp_count[-1]
        tmp_count[0] = count[0] + tmp_count[0]
        tmp_count[1] = count[1] + tmp_count[1]
        tmp[key] = [count[-1],count[0],count[1]]
    
    return tmp, tmp_count

def clusterSumSign(sign_list,feature):
    '''
        给了一个sign列表，对列表进行层次聚类，类别为2，数量多的一类认为是正确的一类，
        对正确多的一类进行求和，返回求和后的值。
    '''

    if feature == True:
        # print('cool')
        sl = [countSign(s)[0] for s in sign_list] #获取-101特征
    else:
        sl = copy.deepcopy(sign_list)

    vec_sign_list = []
    
    for sign in sl:
        tmp_vec=[]
        for v in sign.values():
            if type(v)==list: #说明使用的是-101特征
                tmp_vec.extend(v)
            else:
                tmp_vec.extend(v.view(-1).cpu())
        # tmp = torch.cat(tmp_vec)
        vec_sign_list.append(list(tmp_vec))

    # 进行聚类
    disMat = sch.distance.pdist(vec_sign_list,'cosine') 
    Z=sch.linkage(disMat,method='average') 
    cluster= sch.fcluster(Z, t=2, criterion='maxclust') 
    c1 = np.argwhere(cluster==1).reshape(-1)
    c2 = np.argwhere(cluster==2).reshape(-1)

    if len(c1)>=len(c2):
        idx = c1
    else:
        idx = c2
    
    need_sum_sign = [sign_list[i] for i in idx]
    # print(need_sum_sign)

    return sumWeight(need_sum_sign)


def sameDircWeight(wei1,wei2):

    tmp= {}
    for key,value in wei1.items():
        sign1 = torch.sign(wei1[key])
        sign2 = torch.sign(wei2[key])

        wei1_turn = -1 * value

        wei = torch.where(sign1!=sign2,wei1_turn,value)
        
        tmp[key] = wei

    return tmp

def sortDict(d):
    new_key = sorted(d)
    tmp = {}
    for i in new_key:
        tmp[i] = d[i]
    return tmp

def process_grad_batch(params, clipping=1):
    n = params[0].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    for p in params: 
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_batch *= scaling
        p.grad = torch.mean(p.grad_batch, dim=0)
        p.grad_batch.mul_(0.)

def clipModelGrad(model,clipping=1):
    for p in model.parameters():
        g_norm2 = torch.norm(p.grad,p=2)
        scaling = g_norm2/clipping
        print(scaling)
        p.grad.data = p.grad.data / torch.max(1,scaling)

def boundWeight(server_weight,model_dicts):
    delta_norm = []
    delta_weight = []

    for client_weight in model_dicts.values():
        tmp_delta = {}
        tmp_norm = {}
        for key,value in server_weight.items():
            delta = client_weight[key]-value
            delta_norm_ = torch.norm(delta,p=2)
            tmp_delta[key] = delta
            tmp_norm[key] = delta_norm_
        delta_norm.append(tmp_norm)
        delta_weight.append(tmp_delta)
        # delta_norm.append(torch.norm(weightToVec(subWeight(client_weight,server_weight)),p=2))
        # delta_weight.append(subWeight(client_weight,server_weight))
    
    S = {}
    for key,value in delta_norm[0].items():
        norms = []
        for delta_norm_i in delta_norm:
            norms.append(delta_norm_i[key])

        norms_median = torch.median(torch.tensor(norms))
        S[key] = norms_median
    
    bound_delta_weight=[]

    for i in range(len(delta_weight)):
        tmp = {}
        for key,value in delta_weight[i].items():
            tmp[key] = value / max(1,delta_norm[i][key]/S[key])
        bound_delta_weight.append(tmp)

    return S,bound_delta_weight

def boundWeightAll(server_weight,model_dicts,clipping=None):
    delta_norm = []
    delta_weight = []

    for client_weight in model_dicts.values():
        tmp_delta = {}
        tmp_delta_norm_list = []
        for key,value in server_weight.items():
            # if 'running' in key:
            #     delta = client_weight[key]-value
            #     tmp_delta[key] = delta
            #     continue
            delta = client_weight[key]-value
            tmp_delta[key] = delta
            tmp_delta_norm_list.append(delta.view(-1))
        
        tmp_delta_norm =torch.norm(torch.cat(tmp_delta_norm_list),p=2)

        delta_norm.append(tmp_delta_norm)
        delta_weight.append(tmp_delta)
    
    if clipping != None:
        scale_norm = [dn/clipping for dn in delta_norm]
        bound_delta_weight = []
        for i in range(len(delta_weight)):
            tmp = {}
            for key,value in delta_weight[i].items():
                tmp[key] = value / max(1,scale_norm[i])
            bound_delta_weight.append(tmp)
        return delta_norm,bound_delta_weight
    else:
        S = torch.median(torch.tensor(delta_norm))
        bound_delta_weight = []
        for i in range(len(delta_weight)):
            tmp = {}
            for key,value in delta_weight[i].items():
                # if 'running' in key:
                #     tmp[key] = value
                #     continue
                tmp[key] = value / max(1,delta_norm[i]/S)
            bound_delta_weight.append(tmp)
        return S,bound_delta_weight

# 快速的NMI计算
import numpy as np
import math
def my_NMI(a,b):
    a_n = torch.zeros_like(a,device=a.device)
    a_n[torch.where(a==-1)]=2
    a_n[torch.where(a==0 )]=3
    a_n[torch.where(a==1 )]=5
    b_n = torch.zeros_like(b,device=b.device)
    b_n[torch.where(b==-1)]=11
    b_n[torch.where(b==0 )]=23
    b_n[torch.where(b==1 )]=31

    a_unis = torch.unique(a,return_counts=False).int().cpu().numpy()
    b_unis = torch.unique(b,return_counts=False).int().cpu().numpy()
    # print(a_unis,b_unis)
    # print(a_n)
    total = len(a)
    eps = 1.4e-45

    # c_n_dict = {13:(-1,-1),25:(-1,0),33:(-1,1),14:(0,-1),26:(0,0),34:(0,1),16:(1,-1),18:(1,0),36:(1,1)}
    c_n_dict = {(-1, -1): 13, (-1, 0): 25, (-1, 1): 33, (0, -1): 14, (0, 0): 26, (0, 1): 34, (1, -1): 16, (1, 0): 28, (1, 1): 36}
    c_n = a_n + b_n
    
    # 计算mi
    mi=0
    for a_u in a_unis:
        for b_u in b_unis:
            key = (a_u,b_u)
            value = c_n_dict[key]

            px = 1.0*len(torch.where(a==a_u)[0])/total
            py = 1.0*len(torch.where(b==b_u)[0])/total
            pxy = 1.0*len(torch.where(c_n==value)[0])/total
            # print('key:{},pxy_idx:{}'.format(key,torch.where(c_n==value)[0]))
            mi = mi + pxy*math.log(pxy/(px*py)+eps,2)
    # print(mi)

    # 计算nmi
    hx = 0
    for a_u in a_unis:
        ida_occurCount = 1.0*len(torch.where(a==a_u)[0])
        hx = hx - (ida_occurCount/total)*math.log(ida_occurCount/total+eps,2)
    hy = 0
    for b_u in b_unis:
        idb_occurCount = 1.0*len(torch.where(b==b_u)[0])
        hy = hy - (idb_occurCount/total)*math.log(idb_occurCount/total+eps,2)
    nmi = 2.0*mi/(hx+hy)
    return nmi