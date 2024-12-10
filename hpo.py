import torch as th
import numpy as np

class HyperParameterOptimizer:
    '''
    A hyper parameter optimizer class to optimize the hyperparameters of a simple CNN
    '''
    def __init__(self, hyperparameter, hpspace, method, seed):
        '''
        Inputs :
        - hyperparameter : the hyperparameter we want to optimize (char)
        - hpspace : the possible values for the hyperparameter of study (numpy array)
        - method : the optimization method we want to use (seed)
        - seed : a seed model to start the optimization (torch.nn.Module)
        '''
        self.hyperparameter = hyperparameter
        self.hpspace = hpspace
        self.method = method
        self.module = seed
        print("HyperParameterOptimizer initialized")
    
    def update_hyperparam(new_hp):
        '''
        Update hyperparameter of the optimizer's module
        '''
        pass

    def train_module():
        '''
        Train the optimizers' module
        '''
        pass

    def optimize(self):
        if self.method == 'grid_search':
            hpspace = np.flatten(self.hpspace)
            accuracy = np.zeros(hpspace.shape)
            
            for i in range(len(hpspace)):
                self.update_hyperparam(hpspace[i])
                #################################
                #           TRAIN MODEL         #
                #     EVALUATE MODEL ACCURACY   #
                #################################

            best_hp = np.argmax(accuracy)
            self.result = {
                'Hyper parameter value':best_hp,
            }

            print(f'Hyperparameter optimization finished, best parameter value is %d',self.result['best_hp'])
