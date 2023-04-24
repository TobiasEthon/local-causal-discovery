import numpy as np
import pandas as pd

from causal_discovery.utils import *
from lingam.utils import make_dot

import utils

#%% Settings


n_nodes = [10, 20, 30, 50, 100]
degree = 
graph_level=
weight_range=[0, 5] #I think only relevent for linear methods, Should I use None and then creat lin reg weights randomly ?

method = ['linear', ['nonlinear']]
sem_type = {'linear': ['gauss', 'exp', 'gumble', 'uniform', 'logistic'], 'nonlinear': ['mlp', 'mim']}

n = [100, 1000, 10000, 100000]
noise_scale = [0.1, 1, 10]



#%% Generate graph and sample data

generator = utils.IIDGenerator()

n_nodes = 15
degree=3
graph_level=5
weight_range=[0, 5]

generator.generate_dag(n_nodes, degree, graph_level, weight_range)

#Generate data
method = 'nonlinear'
sem_type = 'mlp'
n = 2000
idx_treatment = 5
noise_scale = 1.0

generator.generate_data(n, method, sem_type, noise_scale)
generator.df


make_dot(generator.dag.transpose(), labels = list(generator.df.columns))


#%% Run algos

generator.define_treatment_variable(6)

generator.optimize()
generator.optimization_result


#%%
sim = utils.Simulator(generator.df.copy())

sim.estimate_parents(method = 'sd')
sim.result
sim.parents

sim.parents_to_vector()
sim.parents_vector  
sim.estimate_counterfactuals([0,1], method = 'lgbm')
sim.counterfactual_Y    

sim.optimize(smoothing = False)
sim.optimization_result


#%% Evaluation

evaluation = utils.EvalMetrics(generator = generator, simulator = sim)
evaluation.graph_metrics()
evaluation.graph_metrics
evaluation.error_counterfactuals([0,1], method = 'lgbm')
evaluation.MSE_counterfactuals
evaluation.MSE_mean_counterfactuals







