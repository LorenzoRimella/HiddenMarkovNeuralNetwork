from BayesianNetwork import muParameter, rhoParameter, LinearBayesianGaussian, BayesianNetwork

import torch
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
    BayesianNetwork1      = BayesianNetwork(dim, BayesianNetwork1_prev)

    check1 = (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() ==  BayesianNetwork1_prev.Linear_layer[0].mu.weight.data.numpy() ).all()

    new_weights = torch.tensor( BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() + 2 )
    BayesianNetwork1.Linear_layer[0].mu.weight.data = new_weights
    check2 = (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() !=  BayesianNetwork1_prev.Linear_layer[0].mu.weight.data.numpy() ).all()

    # print(BayesianNetwork1.Linear_layer[0].mu.weight.shape)
    # print(BayesianNetwork1.Linear_layer[1].rho.bias.shape)

    assert ( check1 and check2)


#
# def test_BayesianNetwork_input():
#
#     dim   = np.array([10, 30, 100])
#
#     BayesianNetwork1_prev = BayesianNetwork(dim)
#     BayesianNetwork1      = BayesianNetwork(dim, BayesianNetwork1_prev)
#
#     x                = torch.tensor( np.random.uniform( 0, 0, (10) ), dtype=torch.float64 ) )
#
#     check1 = (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() ==  BayesianNetwork1_prev.Linear_layer[0].mu.weight.data.numpy() ).all()
#
#     new_weights = torch.tensor( BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() + 2 )
#     BayesianNetwork1.Linear_layer[0].mu.weight.data = new_weights
#     check2 = (BayesianNetwork1.Linear_layer[0].mu.weight.data.numpy() !=  BayesianNetwork1_prev.Linear_layer[0].mu.weight.data.numpy() ).all()
#
#     # print(BayesianNetwork1.Linear_layer[0].mu.weight.shape)
#     # print(BayesianNetwork1.Linear_layer[1].rho.bias.shape)
#
#     assert ( check1 and check2)





# from Refact_tree import branch
#
# from math import sin, cos
# from matplotlib import pyplot as plt
#
# def test_initial_branch():
#
#     b1 = branch(0, 1, 0, 1)
#
#     assert  (b1.x_initial == 0 and b1.y_initial == 0)
#
# def test_plot():
#
#     b1 = branch(0, 1, 0, 1)
#
#     b1.draw_branch()
#
#     plt.show()
