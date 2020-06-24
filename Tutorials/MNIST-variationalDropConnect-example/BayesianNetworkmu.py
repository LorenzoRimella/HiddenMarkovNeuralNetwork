import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import pickle

import numpy as np

####################################################################################
# Define some class for the parameters mu and rho (transformation of sigma)
####################################################################################

class muParameter(nn.Module):

    def __init__(self, in_features, out_features, muParameter_init ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        if muParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/in_features), +np.sqrt(1/in_features), (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -np.sqrt(1/in_features), +np.sqrt(1/in_features), (out_features              ) ), dtype=torch.float64 ) )

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

    def __init__(self, in_features, out_features, rhoParameter_init ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        if rhoParameter_init == False:
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( -5, -4, (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( -5, -4, (out_features             ) ), dtype=torch.float64 ) )

        elif rhoParameter_init == "initial":
            self.weight = nn.Parameter( torch.tensor( np.random.uniform( 0, 0, (out_features, in_features) ), dtype=torch.float64 ) )
            self.bias   = nn.Parameter( torch.tensor( np.random.uniform( 0, 0, (out_features             ) ), dtype=torch.float64 ) )

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
    def __init__(self, mu, rho, p ):
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
        return self.mu * mask.double()  + self.sigma * epsilon.double()



##############################################################################################
# Define a new nn.Linear from a Bayesian point of view which allows an automated reparam trick
##############################################################################################

class LinearBayesianGaussian(nn.Module):

    def __init__(self, in_features, out_features, LinearBayesianGaussian_init, p ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        self.p = p

        if LinearBayesianGaussian_init == False:
            self.mu  = muParameter( in_features, out_features,  muParameter_init = False)
            self.rho = rhoParameter(in_features, out_features, rhoParameter_init = False)

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

        w_stack   = torch.cat( ( self.w_weight.view( self.in_features*self.out_features ), self.w_bias.view( self.out_features ) ), dim = 0 )

        return mu_stack, rho_stack, w_stack


    def performance(self, input):

        return F.linear(input, self.mu.weight, self.mu.bias)


##############################################################################################
# Define a constructor of a Bayesian network
##############################################################################################

class BayesianNetwork(nn.Module):

    def __init__(self, architecture,
                 alpha_k, sigma_k, c,
                 pi, p,
                 BayesianNetwork_init ):

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

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[0], self.architecture[1], BayesianNetwork_init, p = 1) )

            for layer in range(1, self.depth-2):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], BayesianNetwork_init, p = self.p) )

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer+1], self.architecture[layer+2], BayesianNetwork_init, p = 1) )

        elif BayesianNetwork_init == "initial":

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[0], self.architecture[1], BayesianNetwork_init, p = 1) )

            for layer in range(1, self.depth-2):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], "initial", p = self.p) )

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer+1], self.architecture[layer+2], "initial", p = 1) )           

        else:

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[0], self.architecture[1], BayesianNetwork_init.Linear_layer[0], p = 1) )

            for layer in range(1, self.depth-2):
                self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer], self.architecture[layer+1], BayesianNetwork_init.Linear_layer[layer], p = self.p) )

            self.Linear_layer.append( LinearBayesianGaussian( self.architecture[layer+1], self.architecture[layer+2], BayesianNetwork_init.Linear_layer[layer+1], p = 1) )


    def forward(self, x):

        hidden = x

        for layer in range(self.depth-2):
            hidden = self.Linear_layer[layer](hidden)
            hidden = F.relu(hidden)

        output = self.Linear_layer[layer+1](hidden)

        return output

    def performance(self, x):

        hidden = x

        for layer in range(self.depth-2):
            hidden = self.Linear_layer[layer].performance(hidden)
            hidden = F.relu(hidden)

        output = self.Linear_layer[layer+1].performance(hidden)

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

    def get_gaussianlogkernelprior(self, x, mu_prev_grad, sigma_prev_grad, mu_new_grad, p):

        with torch.no_grad():
            mu_new     = mu_new_grad.clone().detach()
            mu_prev    = mu_prev_grad.clone().detach()
            sigma_prev = sigma_prev_grad.clone().detach()

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

        overall = self.pi*p*(f1) + self.pi*(1-p)*(f2) + (1-self.pi)*p*(f3) + (1-self.pi)*(1-p)*(f4)
        summing = (torch.log(overall))

        # print("new")
        # print(f1.sum(), f2.sum(), f3.sum(), f4.sum())

        return summing


    ###############################################################################################
    ###############################################################################################

    def get_gaussianloglikelihood_qw(self, x, mu, sigma, p):

        return -0.5*np.log(2*np.pi) + torch.log( (1-p)/sigma*torch.exp(- (x)*(x)/(2 * sigma*sigma)) + (p)/sigma*torch.exp(- (x - mu)*(x - mu)/(2 * sigma*sigma) ))


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

        split_last = (self.architecture[self.depth-2]*self.architecture[self.depth-1]+self.architecture[self.depth-1])
        split_first = (self.architecture[0]*self.architecture[1]+self.architecture[1])

        w_last   = w[  -split_last:]
        mu_last  = mu[ -split_last:]
        rho_last = rho[-split_last:]
        sigma_last = sigma[-split_last:]
        
        w_first   = w[  0:split_first]
        mu_first  = mu[ 0:split_first]
        rho_first = rho[0:split_first]
        sigma_first = sigma[0:split_first]

        # print(w_last)

        w_bef   = w[  split_first:(len(w)-split_last)]
        mu_bef  = mu[ split_first:(len(w)-split_last)]
        rho_bef = rho[split_first:(len(w)-split_last)]
        sigma_bef = sigma[split_first:(len(w)-split_last)]

        log_qw_theta_last = self.get_gaussianloglikelihood_qw(w_last, mu_last, sigma_last, p = 1)
        log_qw_theta_sum_last = (log_qw_theta_last).sum()
        
        log_qw_theta_first = self.get_gaussianloglikelihood_qw(w_first, mu_first, sigma_first, p = 1)
        log_qw_theta_sum_first = (log_qw_theta_first).sum()

        # print("primo ", log_qw_theta_sum_last)

        log_qw_theta_bef = self.get_gaussianloglikelihood_qw(w_bef, mu_bef, sigma_bef, self.p)
        log_qw_theta_sum_bef = (log_qw_theta_bef).sum()

        log_qw_theta_sum = log_qw_theta_sum_last + log_qw_theta_sum_first + log_qw_theta_sum_bef

        # print("secondo ", log_qw_theta_sum_bef)

        sigma_prev = torch.log1p( torch.exp( rho_prev ) )

        # if alpha_k is zero then set to zero also the prev mu: in this case we do not learn sequentially
        if self.alpha_k == 0:
            with torch.no_grad():
                mu_prev.data.zero_() 
                mu_new.data.zero_() 

        mu_prev_last = mu_prev[-split_last:]
        mu_new_last  = mu_new[-split_last:]
        sigma_prev_last  = sigma_prev[-split_last:]
        
        mu_prev_first = mu_prev[0:split_first]
        mu_new_first = mu_new[0:split_first]
        sigma_prev_first  = sigma_prev[0:split_first]

        mu_prev_bef  = mu_prev[split_first:(len(w)-split_last)]
        mu_new_bef   = mu_new[split_first:(len(w)-split_last)]
        sigma_prev_bef   = sigma_prev[split_first:(len(w)-split_last)]

        log_pw_last     = self.get_gaussianlogkernelprior( w_last , mu_prev_last , sigma_prev_last , mu_new_last, p = 1)
        log_pw_sum_last = (log_pw_last).sum()
        
        log_pw_first     = self.get_gaussianlogkernelprior( w_first , mu_prev_first , sigma_prev_first , mu_new_first, p = 1)
        log_pw_sum_first = (log_pw_first).sum()

        # print("terzo ", log_pw_sum_last)

        log_pw_bef     = self.get_gaussianlogkernelprior( w_bef, mu_prev_bef, sigma_prev_bef, mu_new_bef, self.p)
        log_pw_sum_bef = (log_pw_bef).sum()

        # print("quarto ", log_pw_sum_bef)

        log_pw_sum = log_pw_sum_first + log_pw_sum_last + log_pw_sum_bef

        # print( "new" )
        # print("quinto ", log_qw_theta_sum.data.numpy(), log_pw_sum.data.numpy())

        return (log_qw_theta_sum - log_pw_sum)




##############################################################################################
# Define an Hidden Markov neural network
##############################################################################################


class torchHHMnet(nn.Module):

    # all the default values are derived from MNIST application
    def __init__(self, architecture,
                 alpha_k, sigma_k, c,
                 pi, p,
                 loss_function,
                #  optimizer_choice = optim.Adam,
                 sample_size, minibatch_size, epocs,
                 T, sliding,
                 workers ):

        super().__init__()

        self.architecture  = architecture
        self.depth         = self.architecture.shape[0]

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



    def forward_pass(self, tr_x, tr_y, x_val, y_val, lr_choice, monte_carlo):

        t = 0
        
#         ############################################################################
#         title = "HMMNETcheck-VDropConnect-"
#         title = title+"p-"+str(self.p)
#         title = title+"-pi-"+str(self.pi)
#         title = title+"-sigma-"+str(np.log(self.sigma_k)) 
#         title = title+"-c-"+str(np.log(self.c))

#         title = title+"-lr-"+str(-np.log10(lr_choice) )
#         title = title+"-mc-"+str(monte_carlo)       
#         title = title+"-epochs-"+str(self.epocs)+".txt"       
#         f= open(title,"a")
#         f.close()
#         ############################################################################

        # call the initial model for initialization and so call the stack
        call = self.model_list[t]( torch.tensor(tr_x[0, :], dtype = torch.float64) )

        #create an initial condition with mu different from 0
        initial_cond = BayesianNetwork( self.architecture, self.alpha_k, self.sigma_k, self.c, self.pi, self.p, False )

        mu_prev, rho_prev, w_prev = self.model_list[0].stack()
        mu_new = mu_prev.clone().detach().zero_()

        while t < (self.T):

            t = t+1

            print("Time: ", str(t))
#             string = ["Time: "+ str(t), "\n"]
#             f= open(title,"a")
#             f.writelines(string)
#             f.close()

            new_model = BayesianNetwork( self.architecture, self.alpha_k, self.sigma_k, self.c, self.pi, self.p, initial_cond )

            self.model_list.append(new_model)
            # initial_cond.zero_grad()

            x = tr_x[(t-1)*self.sliding:(t-1)*self.sliding + self.sample_size]
            y = tr_y[(t-1)*self.sliding:(t-1)*self.sliding + self.sample_size]

            tr_x_tensor = torch.tensor( x, dtype = torch.float64)
            tr_y_tensor = torch.tensor( y, dtype = torch.long)

            train = data.TensorDataset(tr_x_tensor, tr_y_tensor)
            train_loader = data.DataLoader(train, batch_size= self.minibatch_size, shuffle=True, num_workers= self.workers)

            iterations = int(self.sample_size/self.minibatch_size)
   
            optimizer   = optim.SGD( self.model_list[t].parameters(), lr = lr_choice, momentum=0.9 )
 
            # set the previous value of mu, rho
            mu_prev, rho_prev, w_prev = self.model_list[t-1].stack()

            for epoch in range(self.epocs):

                string = ["New epoch. "+str(epoch+1), "\n"]
                batch_iter = 0
                for batch in train_loader:
                    # self.model_list[t].zero_grad()
                    optimizer.zero_grad()

                    loss_final_mc = torch.tensor([0.0]).double()

                    for mc in range(monte_carlo):
                        network_output      = self.model_list[t]( batch[0] )
                        loss_network_output = self.loss_function( network_output, batch[1] )

                        loss_prior = ( (1/self.minibatch_size)*(2**(iterations - batch_iter- 1))/(2**(iterations) -1 ) )*self.model_list[t].get_gaussiandistancefromprior(mu_new, mu_prev, rho_prev)

                        loss_final = loss_network_output + loss_prior

                        loss_final_mc = loss_final_mc + loss_final

                    loss_final_mc = (1/monte_carlo)*loss_final_mc
                    loss_final_mc.backward()

                    optimizer.step()
                    batch_iter = batch_iter + 1

                print("Prior "+ str(loss_prior.data.numpy()) + ". Loss "+ str(loss_network_output.data.numpy()))
#                 string = ["Prior "+ str(loss_prior.data.numpy()) + ". Loss "+ str(loss_network_output.data.numpy()), "\n"]
#                 f= open(title,"a")
#                 f.writelines(string)
#                 f.close()

                y_predicted     = np.zeros(len(y_val))
                val_performance =0

                output           = self.model_list[t].performance( torch.tensor( x_val, dtype = torch.float64 ) )
                output_softmax   = F.softmax(output, dim=1)

                y_predicted = np.array( range(0, 10) )[ np.argmax( output_softmax.data.numpy(), 1 ) ]
            
                val_performance = sum(y_val == y_predicted)/len(y_val)
                print("Performance: "+ str(val_performance))
#                 string = ["Performance: "+ str(val_performance), "\n"]
#                 f= open(title,"a")
#                 f.writelines(string)
#                 f.close()

            initial_cond = self.model_list[t]

#             ########################################################################
#             # Save the results

#             HMMNETtime        = {}

#             for time in range(0,t+1):

#                 HMMNETdict        = {}
#                 HMMNETdict['mu']  = {}
#                 HMMNETdict['rho'] = {}

#                 for i in range(0, self.depth-1):
#                     name1 = 'layer'+str(i+1)+'_weight'
#                     name2 = 'layer'+str(i+1)+'_bias'

#                     HMMNETdict['mu'][name1]  = self.model_list[time].Linear_layer[i].mu.weight.data.numpy()
#                     HMMNETdict['mu'][name2]  = self.model_list[time].Linear_layer[i].mu.bias.data.numpy()

#                     HMMNETdict['rho'][name1] = self.model_list[time].Linear_layer[i].rho.weight.data.numpy()
#                     HMMNETdict['rho'][name2]  = self.model_list[time].Linear_layer[i].rho.bias.data.numpy()

#                 HMMNETtime[str(time)] = HMMNETdict


#             title1 = "HMMNETdata-VDropConnect-"
#             title1 = title1+"p-"+str(self.p)
#             title1 = title1+"-pi-"+str(self.pi)
#             title1 = title1+"-sigma-"+str(np.log(self.sigma_k) )
#             title1 = title1+"-c-"+str(np.log(self.c))

#             title1 = title1+"-epochs-"+str(self.epocs)       
#             title1 = title1+"-lr-"+str(-np.log10(lr_choice))
#             title1 = title1+"-mc-"+str(monte_carlo)
#             title1 = title1+".pkl" 
           
#             HMM = open(title1,"wb")
#             pickle.dump(HMMNETtime,HMM)
#             HMM.close()

#             del HMMNETtime
#             ########################################################################




