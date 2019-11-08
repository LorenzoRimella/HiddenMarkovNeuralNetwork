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
            self.bias   = nn.Parameter( torch.tensor( muParameter_init.bias.data.numpy(),   dtype=torch.float64 ) )


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

    # Prior
    def get_gaussianlikelihood(self, x, mu, sigma):

        return (1/(np.sqrt(2*np.pi)*sigma))*torch.exp(-((x-mu)*(x-mu))/(2*sigma*sigma))

    ###############################################################################################

    def get_gaussianlogkernelprior(self, x, mu_prev, sigma_prev, mu_new):
        
        with torch.no_grad():
            mu_new     = torch.tensor( mu_new.data.numpy() )
            mu_prev    = torch.tensor( mu_prev.data.numpy() )
            sigma_prev = torch.tensor( sigma_prev.data.numpy() )
                
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
        mu_prev_check = (self.alpha_k!=0)*mu_prev
        mu_new_check  = (self.alpha_k!=0)*mu_new

        log_pw     = self.get_gaussianlogkernelprior( w, mu_prev_check, sigma_prev, mu_new_check)
        log_pw_sum = (log_pw).sum()

        # print( "new" )
        # print( log_qw_theta_sum, log_pw_sum )

        return (log_qw_theta_sum - log_pw_sum)










##############################################################################################
# Define an Hidden Markov neural network
##############################################################################################


class torchHHMnet(nn.Module):

    # all the default values are derived from MNIST application
    def __init__(self, architecture, 
                 alpha_k = 0.5, sigma_k = np.exp(-1), c = np.exp(7), 
                 pi = 0.5, p = 1.0,
                 loss_function = F.cross_entropy,
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


        initial_model = BayesianNetwork( self.architecture )

        self.model_list = list()
        self.model_list.append(initial_model)



    def forward_pass(self, tr_x, tr_y, x_val, y_val):

        t = 0
        # call the initial model for initialization
        call = self.model_list[t]( torch.tensor(tr_x[0, :], dtype = torch.float64) )

        while t < (self.T):

            t = t+1            

            string = ["Time: "+ str(t), "\n"]
            print(string)

            # the first time step does not depend 
            self.alpha_k = ( self.alpha_k_user )*( t > 1 )

            new_model = BayesianNetwork( self.architecture, self.alpha_k, self.sigma_k, self.c, self.pi, self.p, self.model_list[t-1] )

            self.model_list.append(new_model)   

            x = tr_x[(t-1)*self.sliding:(t-1)*self.sliding + self.sample_size]
            y = tr_y[(t-1)*self.sliding:(t-1)*self.sliding + self.sample_size]
            
            # idx = np.random.choice(range(0, self.sample_size), self.sample_size)
            
            # x = x[idx]
            # y = y[idx]
            
            tr_x_tensor = torch.tensor(x, dtype = torch.float64)
            tr_y_tensor = torch.tensor(np.reshape(y, (np.size(y), 1)), dtype = torch.long)
            train = data.TensorDataset(tr_x_tensor, tr_y_tensor)
            train_loader = data.DataLoader(train, batch_size= self.minibatch_size, shuffle=True, num_workers= self.workers)   

            iterations = int(self.sample_size/self.minibatch_size)

            # optimizer = optimizer_choice(self.model_list[t].parameters())
            optimizer =  optim.Adam(self.model_list[t].parameters())

            # set the previous value of mu, rho
            mu_prev, rho_prev, w_prev = self.model_list[t-1].stack()
            mu_new = torch.tensor( ( ( 1 - 2*self.alpha_k )/( 1 - self.alpha_k ))*mu_prev.data.numpy(), dtype = torch.float64)
                             
            for epoch in range(self.epocs):

                string = ["New epoch. "+str(epoch+1), "\n"]
                print(string)              

                for batch in train_loader:

                    network_output      = self.model_list[t]( batch[0] )
                    loss_network_output = loss_function( network_output, batch[1].squeeze(1) )

                    loss_prior = (1/iterations)*self.model_list[t].get_gaussiandistancefromprior(mu_new, mu_prev, rho_prev)                                          

                    loss_final = loss_network_output + loss_prior

                    loss_final.backward()      

                    optimizer.step()


                y_predicted     = np.zeros(len(y_val))
                val_performance =0

                output           = self.model_list[t]( torch.tensor( x_val ).double() )
                output_softmax   = F.softmax(output, dim=1)

                for tnext in range(0, len( y_val ) ):

                        y_t = np.array( range(0, 10) )[ output_softmax[tnext,:].data.numpy() == max( output_softmax[tnext,:].data.numpy() ) ] 

                        y_predicted[tnext] = y_t

                val_performance = sum(y_val == y_predicted)/len(y_val)
                print("Performance: ", val_performance)                                      


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
















