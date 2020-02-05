import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

import numpy as np

import BayesianNetworkmu1

# set the parameters 
sample_size    = 10000
minibatch_size = 128
epocs          = 600

sliding = 10000
T = 5

###########################################################
# Set the network structure

L = 4
architecture = np.array([784, 400, 400, 10])

alpha_k = 0.25
sigma_k = np.exp(2)
c       = np.exp(10)
p       = 0.3
pi      = 0.5
lr_c = 1e-3
mc_c = 1


########################################################################
# Algorithm training

loss_function = torch.nn.CrossEntropyLoss(reduction = 'sum')

HMMNET = BayesianNetworkmu1.torchHHMnet(architecture, alpha_k, sigma_k, c, pi, p, loss_function, sample_size, minibatch_size, epocs, T, sliding, workers = 4)

x_tr  = np.load(r"ProjectMNIST/x_training.npy")
y_tr  = np.load(r"ProjectMNIST/y_training.npy")

x_val = np.load(r"ProjectMNIST/x_validation.npy")
y_val = np.load(r"ProjectMNIST/y_validation.npy")


torch.manual_seed(2402)
np.random.seed(2402)

HMMNET.forward_pass(x_tr, y_tr, x_val, y_val, lr_c, mc_c)

########################################################################
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
                 
                      
title = "HMMNETdata-grid-mu0-alpha"+str(alpha_k)
title = title+"-p"+str(p)
title = title+"-pi"+str(pi)
title = title+"-sigma"+str(sigma_k) 
title = title+"-c"+str(c)

title = title+"-sample_size"+str(sample_size)  
title = title+"-sliding"+str(sliding)     
title = title+"-minibatch"+str(minibatch_size)
title = title+"-epochs"+str(epocs)      
title = title+"-T"+str(T) 
title = title+"-lr"+str(lr_c)   

HMM = open(title,"wb")
pickle.dump(HMMNETtime,HMM)
HMM.close()






