import torch as th
import numpy as np
import itertools as iter
import LE_NET5_1 as ln5
from model import LeNet5
import copy
from torchsummary import summary
import subprocess

import pso

class HyperParameterOptimizer:
    '''
    A hyper parameter optimizer class to optimize the hyperparameters of a simple CNN
    '''
    def __init__(self, hpspace, seed):
        '''
        Inputs :
        - hpspace : the possible values for the hyperparameter of study (dict of numpy array)
        - method : the optimization method we want to use (seed)
        - seed : a seed model to start the optimization (torch.nn.Module)
        '''
        self.hpspace = hpspace
        self.module = copy.deepcopy(seed)
        self.seed = copy.deepcopy(seed)
        print("HyperParameterOptimizer initialized")

    def load_data(self):
        self.device = (
            "cuda"
            if th.cuda.is_available()
            else "mps"
            if th.backends.mps.is_available()
            else "cpu")
        print(f"Using {self.device} device")

        self.trainloader,self.testloader = ln5.load_data()

        self.module.to(self.device)
        # self.trainloader.to(self.device)
        # self.testloader.to(self.device)

        print("Data loaders have been initialized")
    
    def update_hyperparam(self, new_hp):
        '''
        A function to update the hyperparameter values of the module
        '''
        self.module = LeNet5(new_hp).to(self.device)

    def train_module(self,epochs):
        '''
        Train the optimizers' module
        '''
        criterion = th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam(self.module.parameters(), lr=0.001)

        ln5.train_model(self.module,self.trainloader,criterion,optimizer,self.device,epochs,verbose='partial')
        acc = ln5.test_model(self.module,self.testloader,self.device)

        return acc

    def optimize(self, method, epochs=2, **kwargs):
        '''
        A method to optimize the hyperparameters of the module on a Le-Net 5 architecture
        Name value arguments are available to control the algorithm parameters

        Name value arguments can be fetched through kwargs['PARAMETER NAME']

        Set new hyperparameter config with self.update_hyperparam and train the HPO module
        with this config with self.train_module
        '''

        if method != 'bohb': # no need to compute the hpspace for BOHB method (managed separately)
            hpspace = np.array([dict(zip(self.hpspace.keys(), values)) for values in iter.product(*self.hpspace.values())])

        if method == 'grid_search':
            '''
            Grid Search Optimization

            Tests all combinations of the hyperparameter space to find the best combination
            '''
            accuracy = np.zeros(hpspace.shape)
            
            for i in range(len(hpspace)):
                # self.module = ln5.LeNet5(hpspace[i]).to(self.device)
                self.update_hyperparam(hpspace[i])
                accuracy[i] = self.train_module(epochs)

            best_hp = hpspace[np.argmax(accuracy)]
            cnt = len(hpspace)
            acc = np.max(accuracy)
        
        elif method == 'random_search':
            '''
            Random Search Optimization

            Selects a proportion p of the hyperparameter configurations and tries all the seleted configs
            '''
            assert 'p' in list(kwargs.keys()), "no sampling proportion given at input keyword 'p'"

            p = kwargs['p'] # proportion of samples to try in the hyperparameter optimization
            numel = int(np.fix(p*len(hpspace)))
            ind = np.random.choice(range(len(hpspace)),numel) # choose the hyperparameter configs to try out
            accuracy = np.zeros(ind.shape)

            for i in range(len(ind)):
                print(f"Testing hyperparameter config : {hpspace[ind[i]]}\n")
                self.update_hyperparam(hpspace[ind[i]])
                accuracy[i] = self.train_module(epochs)
            
            best_hp = hpspace[ind[np.argmax(accuracy)]]
            cnt = len(ind)
            acc = np.max(accuracy)

        elif method == 'pso':
            '''
            Particle Swarm Optimization Algorithm

            Runs a particle swarm algorithm to find the best configuration
            '''
            assert 'swarm_size' in list(kwargs.keys()), "no swarm size given at input keyword 'swarm_size'"
            assert 'local_step_size' in list(kwargs.keys()), "no local step size given at input keyword 'local_step_size'"
            assert 'global_step_size' in list(kwargs.keys()), "no swarm size given at input keyword 'global_step_size'"
            assert 'precision' in list(kwargs.keys()), "no stopping criterion given at input keyword 'precision'"
            assert 'inertia' in list(kwargs.keys()), "no inertia coefficient given at input keyword 'inertia'"
            assert 'n_iterations' in list(kwargs.keys()), "no max number of iterations given at input keyword 'n_iterations'"
            
            hpspace_size = np.prod(np.array([len(val) for val in self.hpspace.values()]))
            S = kwargs['swarm_size']
            phi1 = kwargs['local_step_size']
            phi2 = kwargs['global_step_size']
            w = kwargs['inertia']
            n_iterations = kwargs['n_iterations']

            assert w<1 and w>0, "inertia coefficient must be taken between 0 and 1"

            if S>hpspace_size:
                # If the swarm size is bigger than hpspace, we run the equivalent grid search method
                Warning('Swarm size bigger than hpspace, which is equivalent to grid search')
                self.optimize('grid_search')
                
            else :
                x,v = pso.initialize_swarm(S,self.hpspace)
                x_old = x.copy()
                acc_i = np.zeros(len(x)) # to keep track of the accuracy at time i
                cnt = 0

                # Run the first step of the PSO to finish the initialization
                for j in range(S):
                    print(f"PSO initialization  - Particle {j+1}/{S} - Testing hyperparameter config : {x[j]}")
                    self.update_hyperparam(x[j])
                    acc_i[j] = self.train_module(epochs)

                P = pso.Swarm(x[np.argmax(acc_i)]) # Find the best performing config and initialize both local and global maxima
                pi = x.copy()
                accP = np.max(acc_i) # keep track of the accuracy of the global optimum
                accpi = acc_i.copy()

                x = pso.update_swarm(x,v,pi,P,phi1,phi2,w)
                x = x.round(self.hpspace) # make sure the new config is in the hyperparameter space
                
                # Repeat the previous step till convergence and iterate at least once
                while ((x-x_old).global_norm() > np.sqrt(S)*kwargs['precision'] and cnt<n_iterations) or cnt==0 :
                    cnt +=1

                    for j in range(S):
                        print(f"PSO step {cnt} - Particle {j+1}/{S} - Testing hyperparameter config : {x[j]}")
                        self.update_hyperparam(x[j])
                        acc_i[j] = self.train_module(epochs)

                    if np.sum(acc_i>accpi)>0:
                        pi.x[acc_i>accpi] = x.x[acc_i>accpi]
                        accpi[acc_i>accpi] = acc_i[acc_i>accpi]
                    
                    # Update the best global hyperparameter config
                    if np.max(acc_i) > accP:
                        P = pso.Swarm(x[np.argmax(acc_i)])
                        accP = np.max(acc_i)

                    # update the swarm for the next iteration
                    x_old = x.copy()
                    x = pso.update_swarm(x,v,pi,P,phi1,phi2,w)
                    x = x.round(self.hpspace) # make sure the new config is in the hyperparameter space
                
                best_hp = P
                acc = accP

        elif method == 'bohb':
            '''
            Implements the Bayesian Optimization HyperBand algorithm
            The implementation is drawn from the HPBandSter library from:
            Stefan Falkner, Aaron Klein, Frank Hutter. BOHB: Robust and Efficient Hyperparameter Optimization at Scale
            in Proceedings of the 35th International Conference on Machine Learning, PMLR 80:1437-1446, 2018
            https://automl.github.io/HpBandSter/build/html/quickstart.html

            Due to limited time available, we weren't able to provide a cleaner integration than the following : 
            the BOHB library is launched in separate subprocesses to manage parallel trainings and decrease training time
            As the optimizer finishes the computation, it will stay 'stuck' in the terminal, use Ctrl+C to kill the subprocesses
            and extract the results printed in the terminal.
            A 'UnboundLocalError: local variable 'best_hp' referenced before assignment' error will spawn when launching
            the BOHB optimization, please ignore this error

            Hyperparameter space is defined in PyTorchWorker.get_configspace()
            Batch size is defined in PyTorchWorker.__init__()
            Device is defined in PyTorchWorker.compute()
            '''
            assert 'max_epochs' in list(kwargs.keys()), "No maximum number of training epochs given at keyword 'max_epochs'"
            assert 'min_epochs' in list(kwargs.keys()), "No minimum number of training epochs given at keyword 'min_epochs'"
            assert 'n_iterations' in list(kwargs.keys()), "No number of iterations given at keyword 'n_iterations'"
            assert 'n_workers' in list(kwargs.keys()), "No number of workers given at keyword 'n_workers'"

            # launch the master process
            subprocess.Popen(['python','bohb_master.py','--max_budget',str(kwargs['max_epochs']),
                                                          '--min_budget',str(kwargs['min_epochs']),
                                                          '--n_iterations',str(kwargs['n_iterations']),
                                                          '--n_workers',str(kwargs['n_workers'])])

            # launch the slave processes
            for i in range(kwargs['n_workers']):
                subprocess.Popen(['python','bohb_master.py','--max_budget',str(kwargs['max_epochs']),
                                                          '--min_budget',str(kwargs['min_epochs']),
                                                          '--n_iterations',str(kwargs['n_iterations']),
                                                          '--n_workers',str(kwargs['n_workers']),
                                                          '--worker'])

        '''
        End of the different optim methods
        '''
        if method != 'bohb':
            self.result = {
                    'best_hp':str(best_hp),
                    'number of iterations' : cnt,
                    'accuracy' : acc
                }

            return self.result

######################### Test script #########################
if __name__ == '__main__':
    F6Space = np.array([125+i for i in range(200)])
    C1Space = np.array([i+1 for i in range(32)])
    C5Space = np.array([i+40 for i in range(90)])
    HPOptim = HyperParameterOptimizer({'F6':F6Space,'C1_chan':C1Space,'C3_chan':C1Space,'C5_chan':C5Space},seed = LeNet5())
    HPOptim.load_data()
    # res = HPOptim.optimize('bohb',min_epochs = 10,max_epochs = 40, n_iterations = 20, n_workers = 2)
    # res = HPOptim.optimize('pso',epochs=40,swarm_size=5,local_step_size=2,global_step_size=2,precision=1e-5,inertia=0.5,n_iterations=50)
    res = HPOptim.optimize('random_search',epochs=40,p=5.4e-6)
    # res = HPOptim.optimize('grid_search')
    print(res)