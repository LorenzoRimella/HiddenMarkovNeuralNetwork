import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np


###############################################################################################
###############################################################################################

def get_gaussianloglikelihood(x, mu, sigma):

    return (1/(np.sqrt(2*np.pi)*sigma))*torch.exp(-((x-mu)*(x-mu))/(2*sigma*sigma))

###############################################################################################
###############################################################################################


def combined(x, mu_prev, sigma_prev, alpha_k, sigma_k, mu_new, pi, p, c):
    
    with torch.no_grad():
        mu_prev    = torch.tensor(mu_prev.data.numpy())
        sigma_prev = torch.tensor(sigma_prev.data.numpy())
             
    mu1    = mu_new - alpha_k*mu_new + alpha_k*mu_prev
    sigma1 = torch.sqrt(sigma_k*sigma_k + alpha_k*alpha_k*sigma_prev*sigma_prev)
    f1= get_gaussianloglikelihood(x, mu1, sigma1)
             
    mu2    = mu_new - alpha_k*mu_new
    sigma2 = torch.sqrt(sigma_k*sigma_k + alpha_k*alpha_k*sigma_prev*sigma_prev)       
    f2= get_gaussianloglikelihood(x, mu2, sigma2)
             
    mu3    = mu_new - alpha_k*mu_new + alpha_k*mu_prev
    sigma3 = torch.sqrt(sigma_k*sigma_k/(c*c) + alpha_k*alpha_k*sigma_prev*sigma_prev)       
    f3= get_gaussianloglikelihood(x, mu3, sigma3)
             
    mu4    = mu_new - alpha_k*mu_new 
    sigma4 = torch.sqrt(sigma_k*sigma_k/(c*c) + alpha_k*alpha_k*sigma_prev*sigma_prev)   
    f4= get_gaussianloglikelihood(x, mu4, sigma4)

    overall = pi*p*(f1) + pi*(1-p)*(f2) + (1-pi)*p*(f3) + (1-pi)*(1-p)*(f4)
    summing = (torch.log(overall))
    
    # print("old")
    # print(f1.sum(), f2.sum(), f3.sum(), f4.sum())

    return summing


###############################################################################################
###############################################################################################

def get_gaussianloglikelihood_qw(x, mu, sigma, p):

    return -0.5*np.log(2*np.pi) + torch.log( (1-p)/sigma*torch.exp(- (x)*(x)/(2 * sigma*sigma)) + (p)/sigma*torch.exp(- (x - mu)*(x - mu)/(2 * sigma*sigma) ))


##############################################################################################
# Overall likelihood, without the neural network part
##############################################################################################
# Here we are using the previous variational approximation as approximate posterior

def first_likelihood(pi, mu_new, alpha_k, sigma_k, c, model, mu_prev, rho_prev, p, L):

    log_qw_theta_sum = 0
    log_pw_sum       = 0

    # print("old weight")
    # print(model.Linear_layer[0].w_weight)
    
    for i in range(0, L-1):
        sigma_weight = torch.log1p(torch.exp(model.Linear_layer[i].rho.weight))
        sigma_bias   = torch.log1p(torch.exp(model.Linear_layer[i].rho.bias))

        w_weight     = model.Linear_layer[i].w_weight
        w_bias       = model.Linear_layer[i].w_bias

        mu_weight    = model.Linear_layer[i].mu.weight
        mu_bias      = model.Linear_layer[i].mu.bias

        log_qw_theta_weight = get_gaussianloglikelihood_qw(w_weight, mu_weight, sigma_weight, p)
        log_qw_theta_bias = get_gaussianloglikelihood_qw(w_bias, mu_bias, sigma_bias, p)

        log_qw_theta_sum = log_qw_theta_sum + (log_qw_theta_weight).sum()+ (log_qw_theta_bias).sum()


        sigma_prev_weight = torch.log1p(torch.exp(rho_prev[str(i)]["weight"]))
        sigma_prev_bias   = torch.log1p(torch.exp(rho_prev[str(i)]["bias"]))

        mu_prev_weight = mu_prev[str(i)]["weight"]
        mu_prev_bias   = mu_prev[str(i)]["bias"]

        mu_new_weight = mu_new[str(i)]["weight"]
        mu_new_bias   = mu_new[str(i)]["bias"]    

        log_pw_weight     = combined(w_weight, mu_prev_weight, sigma_prev_weight, alpha_k, sigma_k, mu_new_weight, pi, p, c)
        log_pw_bias       = combined(w_bias, mu_prev_bias, sigma_prev_bias, alpha_k, sigma_k, mu_new_bias, pi, p, c)

        log_pw_sum = log_pw_sum + (log_pw_weight).sum() + (log_pw_bias).sum()

    # print( "old" )
    # print( log_qw_theta_sum, log_pw_sum )

    return(log_qw_theta_sum - log_pw_sum)