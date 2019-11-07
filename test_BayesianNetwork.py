from BayesianNetwork import muParameter, rhoParameter, LinearBayesianGaussian

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

    assert ( mu.weight.data.numpy() == mu_prev.weight.data.numpy() ).all()



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
        Linear1.mu.weight.copy_( torch.tensor( np.random.uniform( 5.5, +6, (100, 10) ), dtype=torch.float64 ) )

    output = Linear1( torch.tensor( np.random.uniform( 0, 1, (10) ), dtype=torch.float64 ) )

    print(Linear1.w_weight)


    assert False











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
