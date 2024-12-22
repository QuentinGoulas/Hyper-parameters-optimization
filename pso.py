import torch as th
import numpy as np
import copy

def initialize_swarm(S,hpspace):
    # Initialize the positions of the swarm
    x_ind = np.random.choice(range(len(hpspace)),S,replace=False)
    x = hpspace[x_ind]

    # Initialize at first with 0 speed
    v = np.array([{key:0 for key in hpspace[0].keys()}]*S)

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
        if type(particles) is dict:
            self.x = np.array(particles)
        self.x = particles

    def __add__(self,swarm):
        if len(self)!=1 and len(swarm)!=1:
            assert len(self) == len(swarm), 'Swarms must have the same size or should be broadcastable'

            out = [{key:self.x[i][key]+swarm.x[i][key] for key in list(self.keys())} for i in range(len(self))]
        elif len(self.x)==1:
            out = [{key:self.x[key]+swarm.x[i][key] for key in list(self.keys())} for i in range(len(swarm))]
        else :
            out = (swarm+self).x
        return Swarm(out)
    
    def __sub__(self,swarm):
        if len(self)!=1 and len(swarm)!=1:
            assert len(self.x) == len(swarm.x), 'Swarms must have the same size'

            out = [{key:self.x[i][key]-swarm.x[i][key] for key in list(self.keys())} for i in range(len(self))]
        elif len(self)==1:
            out = [{key:self.x[key]-swarm.x[i][key] for key in list(self.keys())} for i in range(len(swarm))]
        else :
            out = [{key:self.x[i][key]-swarm.x[key] for key in list(self.keys())} for i in range(len(self))]
        return Swarm(out)
    
    def __mul__(self,lamda):
        '''
        /!\ ONLY right-hand scalar multiplication is available
        '''
        assert np.isscalar(lamda), "multiplying factor must be a scalar value"

        out = [{key:self.x[i][key]*lamda for key in list(self.keys())} for i in range(len(self))]
        return Swarm(out)
    
    def __pow__(self,p):
        assert np.isscalar(p), "exponent must be scalar"

        out = [{key:self.x[i][key]**p for key in list(self.keys())} for i in range(len(self))]
        return Swarm(out)
    
    def __len__(self):
        if type(self.x) is dict:
            return 1
        return len(self.x)

    def __str__(self):
        return str(self.x)
    
    def __getitem__(self,slice):
        if type(self.x) is dict:
            return Swarm(self.x)
        return Swarm(self.x[slice])
    
    def keys(self):
        if type(self.x) is dict:
            return self.x.keys()
        return self.x[0].keys()
    
    def copy(self):
        return Swarm(self.x.copy())
    
    def particular_norm(self):
        norms = np.zeros_like(self.x,dtype=np.float64)
        for i in range(len(self)):
            norms[i] = sum(np.array(list(self.x[i].values()))**2)
        
        return np.sqrt(norms)
    
    def global_norm(self):
        norms = self.particular_norm()

        return np.sqrt(np.sum(norms**2,dtype=np.float64))
    
    def round(self, space):
        '''
        Find the closest swarm configuration in a given space by euclidean norm
        '''
        ind = np.zeros_like(self.x,dtype=np.int_)
        for i in range(len(self)):
            dist = (Swarm(self[i])-space).particular_norm()
            ind[i] = np.argmin(dist)
        
        return space[ind]
    
# A test script to validate the swarm object and the PSO functions
if __name__ == '__main__':
    particle1 = np.array([{'F6':80,'C3_chan':6},
                          {'F6':60,'C3_chan':12}])
    particle2 = np.array([{'F6':65,'C3_chan':13},
                          {'F6':79,'C3_chan':6}])

    sw1 = Swarm(particle1)
    sw2 = Swarm(particle2)

    # assert len(sw1)==2, f'wrong length, obtained {len(sw1)}'
    # print(sw1.keys())
    # print(sw1+sw2)
    # print(sw1-sw2)
    # print(sw1*3.5)
    # print(sw1**2)
    print(sw1[1])
    # print(sw1.particular_norm())
    # print(sw1.global_norm())
    # print(sw1.round(sw2))

    # x,v = initialize_swarm(2,np.concatenate((particle1,particle2)))
