import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

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

            # self.weight = nn.Parameter( torch.tensor( np.random.uniform( -0.2, +0.2, (out_features, in_features) ),  dtype=torch.float64 ) )
            # self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -0.2, +0.2, (out_features             ) ), dtype=torch.float64 ) )

        elif muParameter_init == "initial":
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -0.0, +0.0, (out_features, in_features) ),  dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -0.0, +0.0, (out_features             ) ), dtype=torch.float64 ) )           

        else:
            self.weight = nn.Parameter( muParameter_init.weight.clone().detach() )
            self.bias   = nn.Parameter( muParameter_init.bias.clone().detach() )


    def stack(self):

        mu_stack = torch.cat( ( self.weight.view( self.in_features*self.out_features ), self.bias.view( self.out_features ) ), dim=0 )

        return mu_stack


class rhoParameter(nn.Module):

    def __init__(self, in_features, out_features, rhoParameter_init = False ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        if rhoParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -6, -5, (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -6, -5, (out_features             ) ), dtype=torch.float64 ) )

        elif rhoParameter_init == "initial":
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( 1, 1, (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( 1, 1, (out_features             ) ), dtype=torch.float64 ) )

        else:
            self.weight = nn.Parameter( rhoParameter_init.weight.clone().detach() )
            self.bias   = nn.Parameter( rhoParameter_init.bias.clone().detach() )

    def stack(self):

        rho_stack = torch.cat( ( self.weight.view( self.in_features*self.out_features ), self.bias.view( self.out_features ) ), dim=0 )

        return rho_stack
        


##############################################################################################
# Define a class of Gaussian for the reparam trick
##############################################################################################


class Gaussian_dropconnect(object):
    def __init__(self, mu, rho, p = 1):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.p = p

        self.normal    = torch.distributions.Normal(0,1)
        self.bernoulli = torch.distributions.bernoulli.Bernoulli(self.p)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        mask    = self.bernoulli.sample(self.mu.size())
        return self.mu * mask + self.sigma * epsilon



##############################################################################################
# Define a new nn.Linear from a Bayesian point of view which allows an automated reparam trick
##############################################################################################

class LinearBayesianGaussian(nn.Module):

    def __init__(self, in_features, out_features, LinearBayesianGaussian_init = False, p = 1):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        self.p = p

        if LinearBayesianGaussian_init == False:
            self.mu  = muParameter( in_features, out_features)
            self.rho = rhoParameter(in_features, out_features)

        elif LinearBayesianGaussian_init == "initial":
            self.mu  = muParameter( in_features, out_features, LinearBayesianGaussian_init )
            self.rho = rhoParameter(in_features, out_features, LinearBayesianGaussian_init )

        else:
            self.mu  = muParameter( in_features, out_features, LinearBayesianGaussian_init.mu )
            self.rho = rhoParameter(in_features, out_features, LinearBayesianGaussian_init.rho)

        self.w_weight_rv = Gaussian_dropconnect( self.mu.weight, self.rho.weight, self.p )
        self.w_bias_rv   = Gaussian_dropconnect( self.mu.bias  , self.rho.bias,   self.p )     


    def forward(self, input):

        self.w_weight = self.w_weight_rv.sample()
        self.w_bias   = self.w_bias_rv.sample()

        return F.linear(input, self.w_weight, self.w_bias)


    def stack(self):

        mu_stack  = self.mu.stack()
        rho_stack = self.rho.stack()

        w_stack   = torch.cat( ( self.w_weight.view( self.in_features*self.out_features ), self.w_bias.view( self.out_features ) ), dim=0 )

        return mu_stack, rho_stack, w_stack


    def performance(self, input):

        return F.linear(input, self.mu.weight, self.mu.bias)


##############################################################################################
# Define a constructor of a Bayesian network
##############################################################################################

class BayesianNetwork(nn.Module):

    def __init__(self, architecture,
                 alpha_k = 0.5, sigma_k = np.exp(-1), c= np.exp(7),
                 pi = 0.5, p = 1.0,
                 BayesianNetwork_init = False ):

        super().__init__()

        self.architecture  = architecture
        self.depth         = self.architecture.shape[0]

        self.alpha_k = alpha_k
        self.sigma_k = sigma_k
        self.pi      = pi
        self.p       = p
        self.c       = c

        self.Linear_layer  = nn.ModuleList()

        if BayesianNetwork_init == False:

            for layer in range(self.depth-2):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], p = self.p) )

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer+1], self.architecture[layer+2], p = 1) )

        elif BayesianNetwork_init == "initial":
            for layer in range(self.depth-2):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], "initial", p = self.p) )

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer+1], self.architecture[layer+2], "initial", p = 1) )           

        else:
            for layer in range(self.depth-2):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], BayesianNetwork_init.Linear_layer[layer], p = self.p) )

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer+1], self.architecture[layer+2], BayesianNetwork_init.Linear_layer[layer+1], p = 1) )


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

    # Prior
    def get_gaussianlikelihood(self, x, mu, sigma):

        return (1/(np.sqrt(2*np.pi)*sigma))*torch.exp(-((x-mu)*(x-mu))/(2*sigma*sigma))

    ###############################################################################################

    def get_gaussianlogkernelprior(self, x, mu_prev, sigma_prev, mu_new):

        with torch.no_grad():
            mu_new     = mu_new.clone().detach()
            mu_prev    = mu_prev.clone().detach()
            sigma_prev = sigma_prev.clone().detach()

        mu1    = mu_new - self.alpha_k*mu_new + self.alpha_k*mu_prev
        sigma1 = torch.sqrt(self.sigma_k*self.sigma_k + self.alpha_k*self.alpha_k*sigma_prev*sigma_prev)
        f1     = self.get_gaussianlikelihood(x, mu1, sigma1)

        mu2    = mu_new - self.alpha_k*mu_new
        sigma2 = torch.sqrt(self.sigma_k*self.sigma_k + self.alpha_k*self.alpha_k*sigma_prev*sigma_prev)
        f2     = self.get_gaussianlikelihood(x, mu2, sigma2)

        mu3    = mu_new - self.alpha_k*mu_new + self.alpha_k*mu_prev
        sigma3 = torch.sqrt(self.sigma_k*self.sigma_k/(self.c*self.c) + self.alpha_k*self.alpha_k*sigma_prev*sigma_prev)
        f3     = self.get_gaussianlikelihood(x, mu3, sigma3)

        mu4    = mu_new - self.alpha_k*mu_new
        sigma4 = torch.sqrt(self.sigma_k*self.sigma_k/(self.c*self.c) + self.alpha_k*self.alpha_k*sigma_prev*sigma_prev)
        f4     = self.get_gaussianlikelihood(x, mu4, sigma4)

        overall = self.pi*self.p*(f1) + self.pi*(1-self.p)*(f2) + (1-self.pi)*self.p*(f3) + (1-self.pi)*(1-self.p)*(f4)
        summing = (torch.log(overall))

        # print("new")
        # print(f1.sum(), f2.sum(), f3.sum(), f4.sum())

        return summing


    ###############################################################################################
    ###############################################################################################

    def get_gaussianloglikelihood_qw(self, x, mu, sigma):

        return -0.5*np.log(2*np.pi) + torch.log( (1-self.p)/sigma*torch.exp(- (x)*(x)/(2 * sigma*sigma)) + (self.p)/sigma*torch.exp(- (x - mu)*(x - mu)/(2 * sigma*sigma) ))


    ##############################################################################################
    # Overall likelihood, without the neural network part
    ##############################################################################################
    # Here we are using the previous variational approximation as approximate posterior

    def get_gaussiandistancefromprior(self, mu_new, mu_prev, rho_prev):

        log_qw_theta_sum = 0
        log_pw_sum       = 0

        mu, rho, w = self.stack()

        # print("new weight")
        # print(w)

        sigma = torch.log1p( torch.exp( rho ) )

        log_qw_theta = self.get_gaussianloglikelihood_qw(w, mu, sigma)
        log_qw_theta_sum = (log_qw_theta).sum()


        sigma_prev = torch.log1p( torch.exp( rho_prev ) )

        # if alpha_k is zero then set to zero also the prev mu: in this case we do not learn sequentially
        mu_prev_check = (self.alpha_k != 0)*mu_prev
        mu_new_check  = (self.alpha_k != 0)*mu_new

        log_pw     = self.get_gaussianlogkernelprior( w, mu_prev, sigma_prev, mu_new)
        log_pw_sum = (log_pw).sum()

        # print( "new" )
        # print(log_qw_theta_sum.data.numpy(), log_pw_sum.data.numpy())

        return (log_qw_theta_sum - log_pw_sum)










##############################################################################################
# Define an Hidden Markov neural network
##############################################################################################


class torchHHMnet(nn.Module):

    # all the default values are derived from MNIST application
    def __init__(self, architecture,
                 alpha_k = 0.5, sigma_k = np.exp(-1), c = np.exp(7),
                 pi = 0.5, p = 1.0,
                 loss_function = torch.nn.CrossEntropyLoss(reduction = "mean"),
                #  optimizer_choice = optim.Adam,
                 sample_size = 2000, minibatch_size = 128, epocs = 40,
                 T = 300, sliding =100,
                 workers = 3):

        super().__init__()

        self.architecture  = architecture
        self.depth         = self.architecture.shape[0]

        self.alpha_k_user = alpha_k
        self.alpha_k      = alpha_k

        self.sigma_k = sigma_k
        self.pi      = pi
        self.p       = p
        self.c       = c

        # self.optimizer_choice = optimizer_choice
        self.loss_function    = loss_function

        self.sample_size      = sample_size
        self.minibatch_size   = minibatch_size
        self.epocs            = epocs

        self.sliding          = sliding
        self.T                = T

        self.workers = workers


        initial_model = BayesianNetwork( self.architecture, self.alpha_k, self.sigma_k, self.c, self.pi, self.p, "initial" )

        self.model_list = list()
        self.model_list.append(initial_model)



    def forward_pass(self, tr_x, tr_y, x_val, y_val):

        t = 0
        # call the initial model for initialization and so call the stack
        call = self.model_list[t]( torch.tensor(tr_x[0, :], dtype = torch.float64) )

        #create an initial condition with mu different from 0
        initial_cond = BayesianNetwork( self.architecture, self.alpha_k, self.sigma_k, self.c, self.pi, self.p )

        while t < (self.T):

            t = t+1

            string = ["Time: "+ str(t), "\n"]
            print(string)

            # the first time step does not depend
            self.alpha_k = ( self.alpha_k_user )*( t > 1 )
            # print( "alpha_k ", self.alpha_k )

            new_model = BayesianNetwork( self.architecture, self.alpha_k, self.sigma_k, self.c, self.pi, self.p, initial_cond )

            self.model_list.append(new_model)
            initial_cond.zero_grad()

            x = tr_x[(t-1)*self.sliding:(t-1)*self.sliding + self.sample_size]
            y = tr_y[(t-1)*self.sliding:(t-1)*self.sliding + self.sample_size]

            # idx = np.random.choice(range(0, self.sample_size), self.sample_size)

            # x = x[idx]
            # y = y[idx]

            tr_x_tensor = torch.tensor( x, dtype = torch.float64)
            # tr_y_tensor = torch.tensor(np.reshape(y, (np.size(y), 1)), dtype = torch.long)
            tr_y_tensor = torch.tensor( y, dtype = torch.long)

            train = data.TensorDataset(tr_x_tensor, tr_y_tensor)
            train_loader = data.DataLoader(train, batch_size= self.minibatch_size, shuffle=True, num_workers= self.workers)

            iterations = int(self.sample_size/self.minibatch_size)

            # optimizer = optimizer_choice(self.model_list[t].parameters())
            optimizer =  optim.Adam(self.model_list[t].parameters())
            # optimizer   = optim.SGD( self.model_list[t].parameters(), lr = 0.0001) # (1e-2)*(t==1) + (1e-3)*(t!=1) )
 
            # set the previous value of mu, rho
            mu_prev, rho_prev, w_prev = self.model_list[t-1].stack()
            mu_new = ( ( 1 - 2*self.alpha_k )/( 1 - self.alpha_k ))*mu_prev.clone().detach() # mu_prev.clone().detach() # 

            for epoch in range(self.epocs):

                string = ["New epoch. "+str(epoch+1), "\n"]
                print(string)

                for batch in train_loader:
                    # self.model_list[t].zero_grad()
                    optimizer.zero_grad()

                    network_output      = self.model_list[t]( batch[0] )
                    # print(batch[0])
                    loss_network_output = self.loss_function( network_output, batch[1] )
                    # print(batch[1].squeeze(1))

                    loss_prior = (1/iterations)*self.model_list[t].get_gaussiandistancefromprior(mu_new, mu_prev, rho_prev)

                    loss_final = loss_network_output + loss_prior

                    loss_final.backward()

                    optimizer.step()

                print("Prior ", loss_prior.data.numpy(), ". Loss ", loss_network_output.data.numpy())

                y_predicted     = np.zeros(len(y_val))
                val_performance =0

                output           = self.model_list[t]( torch.tensor( x_val ).double() )
                output_softmax   = F.softmax(output, dim=1)

                for tnext in range(0, len( y_val ) ):

                        y_t = np.array( range(0, 10) )[ output_softmax[tnext,:].data.numpy() == max( output_softmax[tnext,:].data.numpy() ) ]

                        y_predicted[tnext] = y_t

                val_performance = sum(y_val == y_predicted)/len(y_val)
                print("Performance: ", val_performance)

            initial_cond = self.model_list[t]


    # ##########################################
    # # Prediction
    # ##########################################

    # def pred(self, x, t):

    #     x_tensor = torch.tensor( x, dtype=torch.float64)

    #     for layer in range(0, self.depth-2):

    #         x_tensor = F.relu( (self.model_list[t].Linear_layer[layer].performance(x_tensor) ) )

    #     x_tensor = self.model_list[t].Linear_layer[self.depth-2].performance(x_tensor)

    #     output_softmax = torch.nn.functional.softmax(x_tensor, dim = 0)

    #     y_t = np.array( range(0, 10) )[ output_softmax.data.numpy()==max(final.data.numpy())]

    #     return({'y': y_t, 'proby': output_softmax.data.numpy()})

















