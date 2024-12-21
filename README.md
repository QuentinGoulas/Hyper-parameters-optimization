# README

### Hyper Parameter Optimizer

This module provides a Hyper Parameter Optimizer framework to optimize a CNN following the Le-Net 5 architecture.

The module uses a HyperParameterOptimizer object (available in hpo.py) with the following methods :
1. load_data : data loading method
2. optimize : run the HPO for a given optimization method

The available methods are : 
1. grid search (method keyword 'grid_search')
2. random search (method keyword 'random_search')
3. (in progress) particle swarm optimization (method keyword 'pso')

To initialize the HyperParameterOptimizer, provide a hyperparameter space with the following dictionary :
{HyperParameter 1 : [HyperParameter 1 values], ...}

The tunable hyperparameters are :
1. 'F6' : size of the final dense layer
2. 'C1_chan','C3_chan','C5_chan' : size of the convolutional layers
3. 'C1_kernel','C3_kernel','C5_kernel' : kernel of the convolutional layers

For now, only the LeNet5 architecture provided in LE_NET5_1.py is available as a seed for the optimizer.

A standard use case is available in hpo.py at the bottom of the script