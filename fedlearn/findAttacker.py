import torch
import numpy as np
import math
from sklearn import metrics

from fedlearn.helper import clusterSumSign


# def distNMI(A,B):
#     return -1*torch.log(torch.tensor(metrics.normalized_mutual_info_score(A, B)))

def distl2(A,B):
    return torch.norm((A-B))  


def distCos(A,B):
    return (1+torch.cosine_similarity(A,B,dim=0))/2

def distNMI(A,B):
    return torch.tensor(metrics.normalized_mutual_info_score(A,B))

def distJoint(A,B,alpha,beta):
    if type(A)!=torch.Tensor:
        A = torch.Tensor(A)
    if type(B)!=torch.Tensor:
        B = torch.Tensor(B)
    if alpha == 0:
        return -1*torch.log(beta*distCos(A,B))
    elif beta == 0:
        return -1*torch.log(alpha*distNMI(A,B))
    else:
        # return -1*torch.log(alpha*distNMI(A,B) + beta*distCos(A,B))
        return -1* (alpha*distNMI(A,B) + beta*distCos(A,B)) + 1

def distEclud(A,B):
    return torch.sqrt(torch.sum(torch.pow((A-B),2)))


def findAttacker(weight_list,rho,alpha,beta):
    print('start find Attacker')
    #获得距离矩阵M
    # print(weight_list[0])
    # M =[]
    # for i in range(len(weight_list)):
    #     t=[]
    #     for j in range(len(weight_list)):
    #         tmp = distNMI(weight_list[i],weight_list[j])
    #         t.append(tmp)
    #     M.append(t)
    W = torch.Tensor(weight_list)
    W = W
    M =np.random.rand(len(W),len(W))
    K=[]
    for i in range(len(W)):
        for j in range(i,len(W)):
            tmp = distJoint(W[i],W[j],alpha,beta)
            M[i][j]=tmp
            if i!=j:
                K.append(tmp)
                M[j][i]=M[i][j]
        # print(t)

    # 使用CFDP
    print('start CFDP:rho{}'.format(rho))
    K_s = np.argsort(K)[math.ceil(len(K)*rho)]
    # print(K)
    pc = K[K_s]
    # 计算密度
    P=[]
    for i in range(len(M)):
        pi = 0
        for j in range(len(M[i])):
            if M[i][j]<pc and i!=j:
                pi = pi + 1
        P.append(pi)
    # 转变为tensor进行计算
    # M = torch.Tensor(M)

    print('densty:',P)

    # 计算密度
    # max_M = torch.max(M)
    # min_M = torch.min(M)
    # P=[]
    # for i in range(len(M)):
    #     pi = 0
    #     for j in range(len(M[i])):
    #         if i!=j and distl2(max_M,M[i][j]) >= distl2(min_M,M[i][j]):
    #             pi = pi+1
    #     P.append(pi)
    
    # 计算密度均值，小于均值为好的client
    P = torch.Tensor(P)
    mean_P = torch.mean(P)
    min_P = torch.min(P)
    ka = mean_P
    # normal_client = np.argwhere(P<=mean_P)
    # attack_client = np.argwhere(P>mean_P)
    # normal_client = np.argwhere(P==0)
    # attack_client = np.argwhere(P>0)
    normal_client = np.argwhere(P<=ka)
    attack_client = np.argwhere(P>ka)

    return normal_client.view(-1),attack_client.view(-1),M



# K-Means

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

def KMeans(W):
    
    W_ten = torch.Tensor(W)
    W_ten.cuda()

    user_function = distJoint
    sample = W_ten

    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    # create K-Means algorithm with specific distance metric
    start_centers = kmeans_plusplus_initializer(sample,2).initialize()
    kmeans_instance = kmeans(sample, start_centers, metric=metric)
    
    # run cluster analysis and obtain results
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    if len(kmeans_instance.get_centers())==1:
        return clusters[0],[]
    c1,c2 = kmeans_instance.get_centers()
    c1,c2 = torch.Tensor(c1),torch.Tensor(c2)

    # 判断有无单一元素的簇
    if len(clusters[0]) == 1:
        return clusters[1],clusters[0]
    elif len(clusters[1]) == 1:
        return clusters[0],clusters[1]
    else:
        # 计算簇到中心的均值判断谁是恶意客户端
        mean_c1 = torch.mean( torch.Tensor( [ distJoint(c1,W_ten[i]) for i in clusters[0]] ) )
        mean_c2 = torch.mean( torch.Tensor( [ distJoint(c2,W_ten[i]) for i in clusters[1]] ) )

        if mean_c1>=mean_c2:
            return clusters[0],clusters[1]
        else:
            return clusters[1],clusters[0]