from BayesianNetwork import muParameter, rhoParameter, LinearBayesianGaussian, BayesianNetwork
from different_prior_computation import first_likelihood

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Test on the parameters

def test_mu_without_initial():

    mu = muParameter(10, 100)

    assert ( mu.weight.data.numpy().shape[1] == 10 and mu.weight.data.numpy().shape[0] == 100 and mu.bias.data.numpy().shape[0] == 100 )



def test_rho_without_initial():

    rho = rhoParameter(10, 100)

    assert ( rho.weight.data.numpy().shape[1] == 10 and rho.weight.data.numpy().shape[0] == 100 and rho.bias.data.numpy().shape[0] == 100 )



def test_mu_with_initial():

    mu_prev = muParameter(10, 100)
    mu      = muParameter(10, 100, mu_prev)

    check1 = ( mu.weight.data.numpy() == mu_prev.weight.data.numpy() ).all()

    mu.weight.data = mu.weight.data + 2
    check2 = ( mu.weight.data.numpy() != mu_prev.weight.data.numpy() ).all()

    assert (check1 and check2)



def test_rho_with_initial():

    rho_prev = rhoParameter(10, 100)
    rho      = rhoParameter(10, 100, rho_prev)

    assert ( rho.weight.data.numpy() == rho_prev.weight.data.numpy() ).all()


def test_stack():

    mu  = muParameter(10, 10 )
    rho = rhoParameter(10, 10 )

    mu_stack  = mu.stack()
    rho_stack = rho.stack()

    (mu_stack.shape == rho_stack.shape)

    assert (mu_stack.shape == rho_stack.shape)



# Test the Linear layer

def test_Linear_without_initial():

    Linear1 = LinearBayesianGaussian(10, 100)

    assert ( ( Linear1.mu.weight.data.numpy().shape[1] == 10 and Linear1.mu.weight.data.numpy().shape[0] == 100 and Linear1.mu.bias.data.numpy().shape[0] == 100 ) and
             ( Linear1.rho.weight.data.numpy().shape[1] == 10 and Linear1.rho.weight.data.numpy().shape[0] == 100 and Linear1.rho.bias.data.numpy().shape[0] == 100 ))



def test_Linear_with_initial():

    Linear1_prev = LinearBayesianGaussian(10, 100)
    Linear1      = LinearBayesianGaussian(10, 100, Linear1_prev)

    assert ( Linear1_prev.rho.weight.data.numpy() == Linear1_prev.rho.weight.data.numpy() ).all()



def test_Linear_w_values():

    Linear1      = LinearBayesianGaussian(10, 100)

    with torch.no_grad():
        Linear1.mu.weight.copy_( torch.tensor( np.random.uniform( 1, 1, (100, 10) ), dtype=torch.float64 ) )

    output = Linear1( torch.tensor( np.random.uniform( 1, 1, (10) ), dtype=torch.float64 ) )

    assert ( output.data.numpy() > 9 ).all() and ( output.data.numpy() < 11 ).all()



def test_Linear_reparam_trick():

    Linear1      = LinearBayesianGaussian(10, 100)

    with torch.no_grad():
        Linear1.mu.weight.copy_( torch.tensor( np.random.uniform( 2, 2, (100, 10) ), dtype=torch.float64 ) )
        #Linear1.rho.weight.copy_( torch.tensor( np.random.uniform( 0, 1, (100, 10) ), dtype=torch.float64 ) )

    output = Linear1( torch.tensor( np.random.uniform( 0, 0, (10) ), dtype=torch.float64 ) )

    loss   = output.sum() + (Linear1.mu.weight*Linear1.mu.weight).sum()
    loss.backward()

    # The derivative of a composition of function in this case is given by all 0 because the inputs are 0
    # The reparam trick add to this derivative the derivative of the loss function wrt mu that is in this case
    # given by all 2*2 (all the mu are 2 and then derive a square function)
    #
    # print(Linear1.mu.weight.grad)

    assert (Linear1.mu.weight.grad.data.numpy() == 4).all()



def test_Linear_multi_input():

    Linear1      = LinearBayesianGaussian(10, 1)

    with torch.no_grad():
        Linear1.mu.weight.copy_( torch.tensor( np.random.uniform( 2, 2, (1, 10) ), dtype=torch.float64 ) )
        Linear1.mu.bias.copy_( torch.tensor( np.random.uniform( 0, 0, (1) ), dtype=torch.float64 ) )
        #Linear1.rho.weight.copy_( torch.tensor( np.random.uniform( 0, 1, (100, 10) ), dtype=torch.float64 ) )

    output = Linear1( torch.tensor( np.random.uniform( 1, 1, (20, 10) ), dtype=torch.float64 ) )

    #print( (output.data.numpy()[0, :] == output.data.numpy()).all() )

    # The derivative of a composition of function in this case is given by all 0 because the inputs are 0
    # The reparam trick add to this derivative the derivative of the loss function wrt mu that is in this case
    # given by all 2*2 (all the mu are 2 and then derive a square function)
    #
    # print(Linear1.mu.weight.grad)

    assert ( (output.data.numpy()[0, :] == output.data.numpy()).all() and output.data.numpy().shape[0] == 20 )



def test_Linear_stack():

    Linear_stack = LinearBayesianGaussian(10, 10)
    output = Linear_stack( torch.tensor( np.random.uniform( 1, 1, (20, 10) ), dtype=torch.float64 ) )

    mu_stack, rho_stack, w_stack = Linear_stack.stack()

    assert ( ( mu_stack.shape == w_stack.shape ) and ( rho_stack.data.numpy().shape[0] == 110 ) )



# Test the BayesianNetwork

def test_BayesianNetwork_without_initial():

    dim   = np.array([10, 30, 100])

    BayesianNetwork1 = BayesianNetwork(dim)

    # print(BayesianNetwork1.Linear_layer[0].mu.weight.shape)
    # print(BayesianNetwork1.Linear_layer[1].rho.bias.shape)

    assert (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy().shape[0] == 30 and
            BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy().shape[1] == 10 and
            BayesianNetwork1.Linear_layer[1].rho.bias.data.numpy().shape[0] == 100)



def test_BayesianNetwork_with_initial():

    dim   = np.array([10, 30, 100])

    BayesianNetwork1_prev = BayesianNetwork(dim)
    BayesianNetwork1      = BayesianNetwork(dim, BayesianNetwork_init = BayesianNetwork1_prev)

    check1 = (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() ==  BayesianNetwork1_prev.Linear_layer[0].mu.weight.data.numpy() ).all()

    new_weights = torch.tensor( BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() + 2 )
    BayesianNetwork1.Linear_layer[0].mu.weight.data = new_weights
    check2 = (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() !=  BayesianNetwork1_prev.Linear_layer[0].mu.weight.data.numpy() ).all()

    # print(BayesianNetwork1.Linear_layer[0].mu.weight.shape)
    # print(BayesianNetwork1.Linear_layer[1].rho.bias.shape)

    assert ( check1 and check2)



def test_BayesianNetwork_input():

    dim   = np.array([10, 30, 10])

    BayesianNetwork_prova_prev = BayesianNetwork(dim)
    BayesianNetwork_prova      = BayesianNetwork(dim, BayesianNetwork_init = BayesianNetwork_prova_prev)

    new_weights = torch.tensor( BayesianNetwork_prova.Linear_layer[1].mu.bias.data.numpy() + 2 )
    BayesianNetwork_prova.Linear_layer[1].mu.bias.data = new_weights

    x = torch.tensor( np.random.uniform( 0, 1, (20, 10) ), dtype= torch.float64 ) 
    y = torch.tensor( np.random.choice(range(0, 10), 20) , dtype= torch.long)

    output_prova = BayesianNetwork_prova(x)

    output_prova_softmax = F.log_softmax(output_prova, 1)
    loss_nll_soft = F.nll_loss(output_prova_softmax, y).data.numpy()

    loss_cross_entr = F.cross_entropy( output_prova, y ).data.numpy()

    check1 = ( loss_cross_entr == loss_nll_soft )


    output_prova_prev = BayesianNetwork_prova_prev(x)

    loss_prova_prev = F.cross_entropy( output_prova_prev, y ).data.numpy()   

    check2 = ( loss_prova_prev != loss_cross_entr )


    assert ( check1 and check2 )




def test_BayesianNetwork_update():

    dim   = np.array([10, 30, 10])

    BayesianNetwork_prova_prev = BayesianNetwork(dim)
    BayesianNetwork_prova      = BayesianNetwork(dim, BayesianNetwork_init = BayesianNetwork_prova_prev)

    # print( BayesianNetwork_prova.Linear_layer[1].mu.weight.data.numpy()[5, :] )
    # print( BayesianNetwork_prova_prev.Linear_layer[1].mu.weight.data.numpy()[5, :] )

    check_equal = (BayesianNetwork_prova.Linear_layer[0].mu.weight.data.numpy() ==  BayesianNetwork_prova_prev.Linear_layer[0].mu.weight.data.numpy() ).all()

    optimizer = optim.Adam(BayesianNetwork_prova.parameters())

    x = torch.tensor( np.random.uniform( 0, 5, (20, 10) ), dtype= torch.float64 ) 
    y = torch.tensor( np.random.choice(range(0, 10), 20) , dtype= torch.long)

    # for iter in range(1):

    output_prova = BayesianNetwork_prova(x)
    loss_prova = F.cross_entropy( output_prova, y )

    loss_prova.backward()
    optimizer.step()

    check_diff = (BayesianNetwork_prova.Linear_layer[1].mu.weight.data.numpy() !=  BayesianNetwork_prova_prev.Linear_layer[1].mu.weight.data.numpy() ).any()

    # print( BayesianNetwork_prova.Linear_layer[1].mu.weight.data.numpy()[5, :] )
    # print( BayesianNetwork_prova_prev.Linear_layer[1].mu.weight.data.numpy()[5, :] )

    assert ( check_equal and check_diff )


def test_BayesianNetwork_stack():

    dim   = np.array([10, 30, 10])

    BayesianNetwork_stack = BayesianNetwork(dim)
    
    x = torch.tensor( np.random.uniform( 0, 5, (20, 10) ), dtype= torch.float64 ) 

    output_prova = BayesianNetwork_stack(x)

    mu_stack, rho_stack, w_stack = BayesianNetwork_stack.stack()

    assert ( ( mu_stack.shape == w_stack.shape ) and ( rho_stack.data.numpy().shape[0] == (10*30+30+30*10+10) ) ) 



def test_BayesianNetwork_prior():

    dim   = np.array([10, 30, 10])

    BayesianNetwork_prova_prev_prev = BayesianNetwork(dim)
    BayesianNetwork_prova_prev      = BayesianNetwork(dim, BayesianNetwork_init = BayesianNetwork_prova_prev_prev)
    BayesianNetwork_prova           = BayesianNetwork(dim, BayesianNetwork_init = BayesianNetwork_prova_prev)

    # print( BayesianNetwork_prova.Linear_layer[1].mu.weight.data.numpy()[5, :] )
    # print( BayesianNetwork_prova_prev.Linear_layer[1].mu.weight.data.numpy()[5, :] )

    optimizer = optim.Adam(BayesianNetwork_prova.parameters())

    x = torch.tensor( np.random.uniform( 0, 5, (20, 10) ), dtype= torch.float64 ) 
    y = torch.tensor( np.random.choice(range(0, 10), 20) , dtype= torch.long)

    # for iter in range(1):
    call         = BayesianNetwork_prova_prev(x)
    output_prova = BayesianNetwork_prova(x)

    loss_network = F.cross_entropy( output_prova, y )

    mu_prev, rho_prev, w_prev = BayesianNetwork_prova_prev.stack()
    loss_prior   = BayesianNetwork_prova.get_gaussiandistancefromprior(mu_prev, mu_prev, rho_prev)

    loss = loss_network + loss_prior

    loss.backward()
    optimizer.step()

    check_diff = True
    for layer in range(0, 1):
        check_diff = ( check_diff and 
                       (BayesianNetwork_prova.Linear_layer[layer].mu.weight.data.numpy() !=  BayesianNetwork_prova_prev.Linear_layer[layer].mu.weight.data.numpy() ).any() )

        check_diff = ( check_diff and 
                       (BayesianNetwork_prova.Linear_layer[layer].rho.weight.data.numpy() !=  BayesianNetwork_prova_prev.Linear_layer[layer].rho.weight.data.numpy() ).any() )

        check_diff = ( check_diff and 
                       (BayesianNetwork_prova.Linear_layer[layer].rho.bias.data.numpy() !=  BayesianNetwork_prova_prev.Linear_layer[layer].rho.bias.data.numpy() ).any() )

        check_diff = ( check_diff and 
                       (BayesianNetwork_prova_prev_prev.Linear_layer[layer].mu.bias.data.numpy() ==  BayesianNetwork_prova_prev.Linear_layer[layer].mu.bias.data.numpy() ).any() )


    # print( BayesianNetwork_prova.Linear_layer[1].mu.weight.data.numpy()[5, :] )
    # print( BayesianNetwork_prova_prev.Linear_layer[1].mu.weight.data.numpy()[5, :] )

    assert ( check_diff )



def test_prior_withdiffcomp():

    dim   = np.array([10, 30, 10])
    L       = 3

    BayesianNetwork_prova_prev      = BayesianNetwork( dim )
    BayesianNetwork_prova           = BayesianNetwork( dim, BayesianNetwork_init = BayesianNetwork_prova_prev)

    x = torch.tensor( np.random.uniform( 0, 5, (20, 10) ), dtype= torch.float64 ) 

    call1 = BayesianNetwork_prova(x)
    call2 = BayesianNetwork_prova_prev(x)

    mu_prev    = {}
    rho_prev   = {}

    with torch.no_grad():
        for i in range(0, L-1):
            mu_i  = {}
            rho_i = {}

            mu_i["weight"] = BayesianNetwork_prova_prev.Linear_layer[i].mu.weight.data.clone().detach()
            mu_i["bias"]   = BayesianNetwork_prova_prev.Linear_layer[i].mu.bias.data.clone().detach()
                                        
            rho_i["weight"]= BayesianNetwork_prova_prev.Linear_layer[i].rho.weight.data.clone().detach()
            rho_i["bias"]  = BayesianNetwork_prova_prev.Linear_layer[i].rho.bias.data.clone().detach()

            mu_prev[str(i)] = mu_i
            rho_prev[str(i)]= rho_i

    pi      = 0.5
    alpha_k = 0.5
    sigma_k = np.exp(0) 
    c       = np.exp(6) 
    model   = BayesianNetwork_prova 
    p       = 1.0
        
    # print( pi, alpha_k, sigma_k, c, p )
    # print( BayesianNetwork_prova.pi, BayesianNetwork_prova.alpha_k, BayesianNetwork_prova.sigma_k, BayesianNetwork_prova.c, BayesianNetwork_prova.p )

    loss_prior_metold  = first_likelihood(pi, mu_prev, alpha_k, sigma_k, c, model, mu_prev, rho_prev, p, L)

    mu_prev2, rho_prev2, w_prev2 = BayesianNetwork_prova_prev.stack()

    loss_prior_metnew   = BayesianNetwork_prova.get_gaussiandistancefromprior(mu_prev2, mu_prev2, rho_prev2)

    # print(loss_prior_metnew - loss_prior_metold)

    assert ( (loss_prior_metnew.data.numpy() - loss_prior_metold.data.numpy())  < np.exp(-5) )



def test_evolution():

    torch.manual_seed(0)
    np.random.seed(0)

    dim   = np.array([10, 30, 10])
    L       = 3

    BayesianNetwork_prev = BayesianNetwork( dim )
              
    BayesianNetwork_1    = BayesianNetwork( dim, BayesianNetwork_init = BayesianNetwork_prev)
    BayesianNetwork_2    = BayesianNetwork( dim, BayesianNetwork_init = BayesianNetwork_prev)

    x = torch.tensor( np.random.uniform( 0, 5, (20, 10) ), dtype= torch.float64 ) 
    y = torch.tensor( np.random.choice( range(0, 10), 20 ), dtype= torch.long )
 
    call1      = BayesianNetwork_1(x)
    call2      = BayesianNetwork_2(x)
    call_prova = BayesianNetwork_prev(x)

    mu_prev    = {}
    rho_prev   = {}

    with torch.no_grad():
        for i in range(0, L-1):
            mu_i  = {}
            rho_i = {}

            mu_i["weight"] = BayesianNetwork_prev.Linear_layer[i].mu.weight.data.clone().detach()
            mu_i["bias"]   = BayesianNetwork_prev.Linear_layer[i].mu.bias.data.clone().detach()
                
            rho_i["weight"]= BayesianNetwork_prev.Linear_layer[i].rho.weight.data.clone().detach()
            rho_i["bias"]  = BayesianNetwork_prev.Linear_layer[i].rho.bias.data.clone().detach()

            mu_prev[str(i)] = mu_i
            rho_prev[str(i)]= rho_i

    pi      = 0.5
    alpha_k = 0.5
    sigma_k = np.exp(0) 
    c       = np.exp(6) 
    model   = BayesianNetwork_1
    p       = 1.0

    check1 = (BayesianNetwork_2.Linear_layer[0].mu.weight.data.numpy() == BayesianNetwork_1.Linear_layer[0].mu.weight.data.numpy()).all()
    print(check1)
        
    # print( pi, alpha_k, sigma_k, c, p )
    # print( BayesianNetwork_prova.pi, BayesianNetwork_prova.alpha_k, BayesianNetwork_prova.sigma_k, BayesianNetwork_prova.c, BayesianNetwork_prova.p )

    optimizer = optim.SGD( BayesianNetwork_1.parameters(), lr = 0.01 )
    optimizer.zero_grad()

    loss_prior_met1  = first_likelihood(pi, mu_prev, alpha_k, sigma_k, c, model, mu_prev, rho_prev, p, L)
    loss_net1        = F.cross_entropy( call1, y)
    loss1 = loss_net1 + loss_prior_met1

    loss1.backward()
    optimizer.step()

    
    mu_prev2, rho_prev2, w_prev2 = BayesianNetwork_prev.stack()
    mu2, rho2, w2 = BayesianNetwork_2.stack()
    # print(mu2)

    loss_prior_met2 = BayesianNetwork_2.get_gaussiandistancefromprior(mu_prev2, mu_prev2, rho_prev2)
    loss_net2       = F.cross_entropy( call2, y )
    loss2 = loss_net2  + loss_prior_met2

    # mu2.grad.zero_()
    # rho2.grad.zero_()
    # w2.grad.zero_()

    loss2.backward()
    # print(BayesianNetwork_2.Linear_layer[0].mu.weight, BayesianNetwork_2.Linear_layer[0].mu.weight.grad)

    BayesianNetwork_2.Linear_layer[0].mu.weight.data  = BayesianNetwork_2.Linear_layer[0].mu.weight.data + 0.01*BayesianNetwork_2.Linear_layer[0].mu.weight.grad.data

    print( BayesianNetwork_1.Linear_layer[0].mu.weight.data.numpy() )
    print( '#############################################################' )
    print( BayesianNetwork_2.Linear_layer[0].mu.weight.data.numpy() )
    
    assert ( BayesianNetwork_2.Linear_layer[0].mu.weight.data.numpy() == BayesianNetwork_1.Linear_layer[0].mu.weight.data.numpy()  ).all()







