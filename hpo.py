import torch as th
import numpy as np
import itertools as iter
import LE_NET5_1 as ln5

class HyperParameterOptimizer:
    '''
    A hyper parameter optimizer class to optimize the hyperparameters of a simple CNN
    '''
    def __init__(self, hpspace, method, seed):
        '''
        Inputs :
        - hpspace : the possible values for the hyperparameter of study (dict of numpy array)
        - method : the optimization method we want to use (seed)
        - seed : a seed model to start the optimization (torch.nn.Module)
        '''
        self.hpspace = hpspace
        self.method = method
        self.module = seed
        self.seed = seed
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
        for hp in self.hyperparameter:
            # Select the node to modify
            node, next_node = self.fetch_node(hp)
            if hp == 'F6':
                node.out_features = new_hp[hp]
                next_node.in_features = new_hp[hp]
            elif hp in ['C1_chan','C3_chan','C5_chan']:
                node.out_channel = new_hp[hp]
                next_node.in_channel = new_hp[hp]
            elif hp == ['C1_kernel','C3_kernel','C5_kernel']:
                node.kernel_size = new_hp[hp]
            pass

    def fetch_node(self, hp):
        if hp == 'F6':
            node = self.module.fc1
            next_node = self.module.fc2
        elif hp in ['C1_chan','C1_kernel']:
            node = self.module.conv1
            next_node = self.module.conv2
        elif hp in ['C3_chan','C3_kernel']:
            node = self.module.conv2
            next_node = self.module.conv3
        elif hp in ['C5_chan','C5_kernel']:
            node = self.module.conv5
            next_node = self.module.fc1
        else :
            raise NameError('Model node unknown')
        
        return node, next_node

    def train_module(self,epochs):
        '''
        Train the optimizers' module
        '''
        criterion = th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam(self.module.parameters(), lr=0.001)

        ln5.train_model(self.module,self.trainloader,criterion,optimizer,self.device,epochs)
        acc = ln5.test_model(self.module,self.testloader,self.device)

        return acc

    def optimize(self,**kwargs):
        if self.method == 'grid_search':
            hpspace = (dict(zip(self.hpspace.keys(), values)) for values in iter.product(*self.hpspace.values()))
            accuracy = np.zeros(hpspace.shape)
            epochs = 2
            
            for i in range(len(hpspace)):
                self.module = self.seed # always start the optimization from the seed
                self.update_hyperparam(hpspace[i])
                accuracy[i] = self.train_module(epochs)

            best_hp = self.hpspace[np.argmax(accuracy)]
            self.result = {
                'best_hp':best_hp,
            }

            print(f"Hyperparameter optimization finished, best parameter value is {self.result['best_hp']}")
        
        if self.method == 'random_search':
            p = kwargs['p'] # proportion of samples to try in the hyperparameter optimization
            hpspace = (dict(zip(self.hpspace.keys(), values)) for values in iter.product(*self.hpspace.values()))
            ind = np.random.uniform(0,len(hpspace),np.floor(p*len(hpspace))) # choose the hyperparameter configs to try out
            accuracy = np.zeros(ind.shape)
            epochs = 2

            for i in ind:
                self.module = self.seed # always start the optimization from the seed
                self.update_hyperparam(hpspace[i])
                accuracy[i] = self.train_module(epochs)
            
            best_hp = self.hpspace[ind[np.argmax(accuracy)]]
            self.result = {
                'best_hp':best_hp
            }

######################### Test script #########################
if __name__ == '__main__':
    HPOptim = HyperParameterOptimizer('dense',np.array([10, 50, 84, 160]),'grid_search',seed = ln5.LeNet5())
    HPOptim.load_data()
    HPOptim.optimize()