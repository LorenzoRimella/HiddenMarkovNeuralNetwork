import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

####################################################################################
# Define some class for the parameters mu and rho (transformation of sigma)
####################################################################################

class muParameter(nn.Module):

    def __init__(self, in_features, out_features, muParameter_init = False ):
        super().__init__()

        if muParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/in_features), +np.sqrt(1/in_features), (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/in_features), +np.sqrt(1/in_features), (out_features              ) ), dtype=torch.float64 ) )

        else:
            self.weight = nn.Parameter( muParameter_init.weight )
            self.bias   = nn.Parameter( muParameter_init.bias   )



class rhoParameter(nn.Module):

    def __init__(self, in_features, out_features, rhoParameter_init = False ):
        super().__init__()

        if rhoParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -4, -5, (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -4, -5, (out_features              ) ), dtype=torch.float64 ) )

        else:
            self.weight = nn.Parameter( rhoParameter_init.weight )
            self.bias   = nn.Parameter( rhoParameter_init.bias   )

##############################################################################################
# Define a new nn.Linear from a Bayesian point of view which allows an automated reparam trick
##############################################################################################

class LinearBayesianGaussian(nn.Module):

    def __init__(self, in_features, out_features, LinearBayesianGaussian_init = False):
        super().__init__()

        if LinearBayesianGaussian_init == False:
            self.mu  = muParameter( in_features, out_features)
            self.rho = rhoParameter(in_features, out_features)

        else:
            self.mu  = muParameter( in_features, out_features, LinearBayesianGaussian_init.mu )
            self.rho = rhoParameter(in_features, out_features, LinearBayesianGaussian_init.rho)

    def forward(self, input):

        sigma_weight   = torch.log1p(torch.exp(self.rho.weight))
        epsilon_weight = torch.distributions.Normal(0,1).sample( self.mu.weight.size() )
        w_weight       = self.mu.weight + sigma_weight * epsilon_weight

        sigma_bias   = torch.log1p(torch.exp(self.rho.bias))
        epsilon_bias = torch.distributions.Normal(0,1).sample( self.mu.bias.size() )
        w_bias       = self.mu.bias + sigma_bias * epsilon_bias

        return F.linear(input, w_weight, w_bias)




class torchnet(nn.Module):

  def __init__(self, L, dim, model = False ):

    super().__init__()

    self.L    = L
    self.dim  = dim

    self.mu_weight  = nn.ModuleList()
    self.mu_bias    = nn.ModuleList()

    self.rho_weight = nn.ModuleList()
    self.rho_bias   = nn.ModuleList()


    if model == False:

      for i in range(0, self.L-1):
        mu_i_weight  = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/dim[i]), +np.sqrt(1/dim[i]), (dim[i+1], dim[i]) ), dtype=torch.float64 ) )
        rho_i_weight = nn.Parameter( torch.tensor( np.random.uniform( -4, +5, (dim[i+1], dim[i])                                 ), dtype=torch.float64 ) )

        mu_i_bias  = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/dim[i]), +np.sqrt(1/dim[i]), (dim[i+1]) ), dtype=torch.float64 ) )
        rho_i_bias = nn.Parameter( torch.tensor( np.random.uniform( -4, +5, (dim[i+1])                                 ), dtype=torch.float64 ) )

        self.mu_weight.append(mu_i_weight)
        self.mu_bias.append(mu_i_bias)

        self.rho.append(rho_i)
