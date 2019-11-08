import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


####################################################################################
# Define some class for the parameters mu and rho (transformation of sigma)
####################################################################################

class muParameter(nn.Module):

    def __init__(self, in_features, out_features, muParameter_init = False ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        if muParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/in_features), +np.sqrt(1/in_features), (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/in_features), +np.sqrt(1/in_features), (out_features              ) ), dtype=torch.float64 ) )

        else:
            self.weight = nn.Parameter( torch.tensor( muParameter_init.weight.data.numpy(), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( muParameter_init.bias.data.numpy(),   dtype=torch.float64 )   )


    def stack(self):

        mu_stack = torch.cat( ( self.weight.view( self.in_features*self.out_features ), self.bias.view( self.out_features ) ), dim=0 )

        return mu_stack


class rhoParameter(nn.Module):

    def __init__(self, in_features, out_features, rhoParameter_init = False ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        if rhoParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -4, -5, (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -4, -5, (out_features              ) ), dtype=torch.float64 ) )

        else:
            self.weight = nn.Parameter( torch.tensor( rhoParameter_init.weight.data.data.numpy(), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( rhoParameter_init.bias.data.data.numpy(), dtype=torch.float64 )   )
    
    def stack(self):

        rho_stack = torch.cat( ( self.weight.view( self.in_features*self.out_features ), self.bias.view( self.out_features ) ), dim=0 )

        return rho_stack


##############################################################################################
# Define a new nn.Linear from a Bayesian point of view which allows an automated reparam trick
##############################################################################################

class LinearBayesianGaussian(nn.Module):

    def __init__(self, in_features, out_features, LinearBayesianGaussian_init = False):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        if LinearBayesianGaussian_init == False:
            self.mu  = muParameter( in_features, out_features)
            self.rho = rhoParameter(in_features, out_features)

        else:
            self.mu  = muParameter( in_features, out_features, LinearBayesianGaussian_init.mu )
            self.rho = rhoParameter(in_features, out_features, LinearBayesianGaussian_init.rho)


    def forward(self, input):

        sigma_weight   = torch.log1p(torch.exp(self.rho.weight))
        epsilon_weight = torch.distributions.Normal(0,1).sample( self.mu.weight.size() )
        self.w_weight       = self.mu.weight + sigma_weight * epsilon_weight

        sigma_bias   = torch.log1p(torch.exp(self.rho.bias))
        epsilon_bias = torch.distributions.Normal(0,1).sample( self.mu.bias.size() )
        self.w_bias       = self.mu.bias + sigma_bias * epsilon_bias

        return F.linear(input, self.w_weight, self.w_bias)


    def stack(self):

        mu_stack  = self.mu.stack()
        rho_stack = self.rho.stack()

        w_stack   = torch.cat( ( self.w_weight.view( self.in_features*self.out_features ), self.w_bias.view( self.out_features ) ), dim=0 )

        return mu_stack, rho_stack, w_stack



##############################################################################################
# Define a constructor of a Bayesian network 
##############################################################################################

class BayesianNetwork(nn.Module):

    def __init__(self, architecture, BayesianNetwork_init = False ):

        super().__init__()

        self.architecture  = architecture
        self.depth         = self.architecture.shape[0]

        self.Linear_layer  = nn.ModuleList()

        if BayesianNetwork_init == False:

            for layer in range(self.depth-1):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1]) )

        else:
            for layer in range(self.depth-1):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], BayesianNetwork_init.Linear_layer[layer]) )


    def forward(self, x):

        hidden = x

        for layer in range(self.depth-2):
            hidden = self.Linear_layer[layer](hidden)
            hidden = F.relu(hidden)

        output = self.Linear_layer[layer+1](hidden)

        return output

    def stack(self):

        mu_stack, rho_stack, w_stack = self.Linear_layer[0].stack()

        for layer in range(1, self.depth-1):
            mu_stack_new, rho_stack_new, w_stack_new = self.Linear_layer[layer].stack()

            mu_stack  = torch.cat( ( mu_stack,  mu_stack_new  ) , dim=0 )
            rho_stack = torch.cat( ( rho_stack, rho_stack_new ) , dim=0 )
            w_stack   = torch.cat( ( w_stack,   w_stack_new   ) , dim=0 )

        return mu_stack, rho_stack, w_stack



##############################################################################################
# Define an Hidden Markov neural network
##############################################################################################

# Collection of useful functions
###############################################################################################

def get_gaussianlikelihood(x, mu, sigma):

    return (1/(np.sqrt(2*np.pi)*sigma))*torch.exp(-((x-mu)*(x-mu))/(2*sigma*sigma))

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
    
    for i in range(0, L-1):
        sigma_weight = torch.log1p(torch.exp(model.rho[i].weight))
        sigma_bias   = torch.log1p(torch.exp(model.rho[i].bias))

        w_weight     = model.w[i].weight
        w_bias       = model.w[i].bias

        mu_weight    = model.mu[i].weight
        mu_bias      = model.mu[i].bias

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

    return(log_qw_theta_sum - log_pw_sum)
































