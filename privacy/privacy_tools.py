import numpy as np
import torch
import copy
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getGaussNoise(weight,sigma,S = None):
    tmp = {}

    if S != None:
        pass

    else:
        for key,value in weight.items():
            lay_noise = torch.normal(0,sigma,value.shape).to(device)
            tmp[key] = lay_noise
    
    return tmp

def getDpLr(weight,lr,sigma):
    tmp = {}
    for key,value in weight.items():
        noise = torch.normal(0,sigma,value.shape).to(device)
        noise_max = torch.max(noise)
        noise_min = torch.min(noise)
        noise_normal = (noise-noise_min)/noise_max
        tmp[key] = lr* noise_normal
    return tmp