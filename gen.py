import numpy as np
import copy
from pso import Swarm

class Individu:
    def __init__(self, config=[]):
        self.config = config

    def random_config(self, hpspace):
        config_ind = np.random.choice(range(len(hpspace)),replace=False)
        self.config = hpspace[config_ind]

    def keys(self):
        return self.config.keys()
    
    def copy(self):
        return Individu(self.config.copy())

def initialize_population(hpspace, P):
    population = [Individu() for _ in range(P)]
    for indiv in population:
        indiv.random_config(hpspace)
    return population

def crossover(par1, par2, cp):
    assert list(par1.keys()) == list(par2.keys()), "parents must have the same dictionary for crossover"
    
    child1 = par1.copy()
    child2 = par2.copy()
    n = len(par1)
    for key in list(par1.keys()):
        if np.random.rand() < cp:
            child1.config[key] = par2[key]
            child2.config[key] = par1[key]
    return child1, child2

def mutation(indiv, mp, hpspace):
    for key in list(indiv.keys()):
        if np.random.rand() < mp:
            indiv[key] = hpspace[np.random.choice(range(len(hpspace)))]
    return indiv

def select_population(population, accuracies, bp, lp):
    S = len(population)
    sorted_indices = np.argsort(accuracies)[::-1]
    num_bp = int(S*bp)
    num_lp = int(S*lp)
    selected_population = population[sorted_indices[:bp]] 
    rest_population = population[sorted_indices[bp:]]
    random_lp_indices = np.random.choice(range(len(rest_population)), num_lp, replace=False)
    selected_population = np.concatenate((selected_population, rest_population[random_lp_indices]))
    return selected_population
     


    
# A test script to validate the swarm object and the PSO functions
if __name__ == '__main__':
    pass