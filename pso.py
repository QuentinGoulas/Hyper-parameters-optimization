import torch as th
import numpy as np
import copy

def initialize_swarm(S,hpspace):
    # Initialize the positions of the swarm
    x_ind = np.random.choice(range(len(hpspace)),S)
    x = hpspace[x_ind]

    # Initialize at first with 0 speed
    v = np.array([dict(zip(hpspace[0].keys,0))]*S)

    return Swarm(x),Swarm(v)

def update_swarm(x,v,p,P,phi1,phi2):
    u1 = np.random.uniform(0,phi1)
    u2 = np.random.uniform(0,phi2)

    v = v + (p-x)*u1 + (P-x)*u2
    x = x + v

    return x

class Swarm:
    '''
    A Swarm class to perform classic algebraic operations on a swarm (pretty much )
    '''
    def __init__(self, particles):
        Swarm.x = particles

    def __add__(self,swarm):
        if len(self.x)!=1 & len(swarm.x!=1):
            assert len(self.x) == len(swarm.x), 'Swarms must have the same size'

            out = [{key:self.x[i][key]+swarm.x[i][key] for key in list(self.x.keys())} for i in range(len(self.x))]
        elif len(self.x)!=1:
            out = [{key:self.x[key]+swarm.x[i][key] for key in list(self.x.keys())} for i in range(len(swarm.x))]
        else :
            out = (swarm+self).x
        return Swarm(out)
    
    def __sub__(self,swarm):
        if len(self.x)!=1 & len(swarm.x!=1):
            assert len(self.x) == len(swarm.x), 'Swarms must have the same size'

            out = [{key:self.x[i][key]-swarm.x[i][key] for key in list(self.x.keys())} for i in range(len(self.x))]
        elif len(self.x)!=1:
            out = [{key:self.x[key]-swarm.x[i][key] for key in list(self.x.keys())} for i in range(len(swarm.x))]
        else :
            out = (swarm-self).x
        return Swarm(out)
    
    def __mul__(self,lamda):
        '''
        /!\ ONLY right-hand scalar multiplication is available
        '''
        assert np.isscalar(lamda), "multiplying factor must be a scalar value"

        out = [{key:self.x[i][key]*lamda for key in list(self.x.keys())} for i in range(len(self.x))]
        return Swarm(out)
    
    def __pow__(self,p):
        assert np.isscalar(p), "exponent must be scalar"

        out = [{key:self.x[i][key]**p for key in list(self.x.keys())} for i in range(len(self.x))]
        return Swarm(out)
    
    def __len__(self):
        return len(self.x)
    
    def particular_norm(self):
        norms = np.zeros_like(self.x)
        for i in range(len(self.x)):
            norms[i] = sum(np.array(list(self.x.values()))**2)
        
        return np.sqrt(norms)
    
    def global_norm(self):
        norms = self.particular_norm()

        return np.sqrt(np.sum(norms**2))
    
    def round(self, space):
        '''
        Find the closest swarm configuration in a given space by euclidean norm
        '''
        ind = np.zeros_like(self.x)
        for i in range(len(self)):
            dist = (self-space).particular_norm()
            ind[i] = np.argmin(dist)
        
        return space[ind]
