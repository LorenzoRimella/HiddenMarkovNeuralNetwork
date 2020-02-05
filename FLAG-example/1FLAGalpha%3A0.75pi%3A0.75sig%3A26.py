https://www.cc.gatech.edu/cpl/projects/videotexture/SIGGRAPH2000/index.htm

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

import numpy as np

torch.manual_seed(123)
np.random.seed(123)

#################################################
# Load the data

x = np.load(r'ProjectFLAG/xFLAG.npy')
y = np.load(r'ProjectFLAG/yFLAG.npy')

###########################################################
# import the module
import BayesianNetwork

# set the parameters 
sample_size    = 36
minibatch_size = 36
epocs          = 150

sliding = 1
T = 300

###########################################################
# Set the network structure
architecture = np.array([130, 500, 20, 500, 130])
L = 5

pi      = 0.75
p       = 1.0
alpha_k = 0.75
sigma_k = np.exp(2)
c       = np.exp(8)


########################################################################
# Algorithm training

loss_function = torch.nn.MSELoss(reduction = "sum")

HMMNET = BayesianNetwork.torchHHMnet(architecture, alpha_k, sigma_k, c, pi, p, loss_function, sample_size, minibatch_size, epocs, T, sliding, workers = 1)

HMMNET.forward_pass(x, y, 1e-4)

#################################################
# Save the results

HMMNETtime        = {}

for t in range(0,T+1):

    HMMNETdict        = {}
    HMMNETdict['mu']  = {}
    HMMNETdict['rho'] = {}

    for i in range(0,L-1):
        name1 = 'layer'+str(i+1)+'_weight'
        name2 = 'layer'+str(i+1)+'_bias'

        HMMNETdict['mu'][name1]  = HMMNET.model_list[t].Linear_layer[i].mu.weight.data.numpy()
        HMMNETdict['mu'][name2]  = HMMNET.model_list[t].Linear_layer[i].mu.bias.data.numpy()

        HMMNETdict['rho'][name1] = HMMNET.model_list[t].Linear_layer[i].rho.weight.data.numpy()
        HMMNETdict['rho'][name2]  = HMMNET.model_list[t].Linear_layer[i].rho.bias.data.numpy()
        
    HMMNETtime[str(t)] = HMMNETdict
                 
                      
title = "HMMNETFLAGdata-alpha"+str(alpha_k)
title = title+"-p"+str(p)
title = title+"-pi"+str(pi)
title = title+"-sigma"+str(sigma_k) 
title = title+"-c"+str(c)

title = title+"-sample_size"+str(sample_size)  
title = title+"-sliding"+str(sliding)     
title = title+"-minibatch"+str(minibatch_size)
title = title+"-epochs"+str(epocs)      
title = title+"-T"+str(T)    

HMM = open(title,"wb")
pickle.dump(HMMNETtime,HMM)
HMM.close()


HMM = open(title,"wb")
pickle.dump(HMMNETtime,HMM)
HMM.close()


