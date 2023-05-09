import numpy as np
import pandas as pd

from lingam.utils import make_dot

import utils
import time

#%% Settings


n_nodes = [5]#, 20, 30, 50, 100
degree = [3]#8
#graph_level=[5, 10, 15]
weight_range=[[0, 5]] #I think only relevent for linear methods, Should I use None and then creat lin reg weights randomly ?

method = ['nonlinear']#['linear', 'nonlinear']
sem_type = {'linear': ['gauss', 'exp', 'gumble', 'uniform', 'logistic'], 'nonlinear': ['mlp']}#, 'mim'

N = [10000]#[100, 1000, 10000, 100000]
noise_scale = [1.]#[0.1, 1, 10]

method_sim = ['ldecc']#, 'ldecc', 'mb_by_mb'
method_fct = ['lgbm']

#%% Run through loop

graph_metrics = {m: [] for m in method_sim}
mse_y = {m: [] for m in method_sim}
mse_mean = {m: [] for m in method_sim}

t0 = time.time()

for n_node in n_nodes:
    for deg in degree:
        for graph_l in [int(n_node/1)]:#, int(n_node/2), int(n_node)#graph_level
            for weight_r in weight_range:
                #Setup the generator
                generator = utils.IIDGenerator()
                generator.generate_dag(n_node, deg, graph_l, weight_r)
                dot = make_dot(generator.dag.transpose(), labels = ['x_' + str(i) for i in range(generator.dag.shape[0]-1)] + ['Y'])
                dot.render('dag_plots/dag' + '_nodes_' + str(n_node) + '_degree_' + str(deg) +'_graph_level_' + str(graph_l))

                for met in method:
                    for sem_t in sem_type[met]:
                        for n in N:
                            for noise_s in noise_scale:
                                generator.generate_data(n, met, sem_t, noise_s)
                                idx_treatment = utils.choose_idx_treatment(generator.dag)
                    
                                for idx_t in [idx_treatment[0]]:
                                    print(idx_t)
                                    generator.define_treatment_variable(idx_t)
                                        
                                    for met_fct in method_fct:
                                        for met_sim in method_sim:  
                                            #Setup the simulator
                                            sim = utils.Simulator(generator.df.copy())
                                            sim.estimate_parents(method = met_sim)
                                            print('Estimated parents with', met_sim)
                                            print(sim.parents)
                                            #print(sim.parents_all)
# =============================================================================
#                                             X_int = sim.df.iloc[:, idx_t].quantile(0.66)
#                                             
#                                             #Setup Evaluation
#                                             evaluation = utils.EvalMetrics(generator = generator, simulator = sim)
#                                             evaluation.graph_metrics()
#                                             graph_metrics[met_sim].append(evaluation.graph_metrics)
#                                             evaluation.error_counterfactuals(X_int, method = met_fct)
#                                             mse_y[met_sim].append(evaluation.MSE_counterfactuals)
#                                             mse_mean[met_sim].append(evaluation.MSE_mean_counterfactuals)
#                                             del sim
#                                             del evaluation
# 
# =============================================================================
t1 = time.time()

print('This simulation took: ', t1-t0)

#%%

res = {m: [] for m in method_sim}                                    
for m in graph_metrics.keys():
    for d in graph_metrics[m]:
        v = list(map(np.mean, d.values()))
        k = d.keys()
        t = dict(zip(k, v))
        res[m].append(t)
        
    res[m] = pd.DataFrame(res[m])
    res[m] = dict(res[m].mean())
    

res_mse = {}
for m in mse_y.keys():
    res_mse[m] = np.mean(list(map(lambda x: np.mean(x), mse_y[m])))

res_mean = {}
for m in mse_mean.keys():
    res_mean[m] = np.mean(list(map(lambda x: np.mean(x), mse_mean[m])))




#%%
N_OBS = 1000
N_NODES = 20 
DEGREE = 4
GRAPH_LEVEL = 20
METHOD = 'nonlinear'
SEM_TYPE = 'mlp'
WEIGHT = [0, 5]
NOISE_SCALE = 1.0


generator = utils.IIDGenerator()
generator.generate_dag(N_NODES, DEGREE, GRAPH_LEVEL, WEIGHT)
generator.generate_data(N_OBS, METHOD, SEM_TYPE, NOISE_SCALE)


#%% Generate graph and sample data

generator = utils.IIDGenerator()

# =============================================================================
# n_nodes = 15
# degree=3
# graph_level=5
# weight_range=[0, 5]
# 
# =============================================================================
generator.generate_dag(n_nodes, degree, graph_level, weight_range)

# =============================================================================
# #Generate data
# method = 'nonlinear'
# sem_type = 'mlp'
# n = 2000
# idx_treatment = 5
# noise_scale = 1.0
# =============================================================================

generator.generate_data(n, method, sem_type, noise_scale)
#generator.df


#make_dot(generator.dag.transpose(), labels = list(generator.df.columns))


#%% Run algos

generator.define_treatment_variable(6)

#generator.optimize()
#generator.optimization_result


#%%
sim = utils.Simulator(generator.df.copy())

sim.estimate_parents(method = 'sd')
#sim.result
#sim.parents

sim.parents_to_vector()
#sim.parents_vector  
sim.estimate_counterfactuals([0,1], method = 'lgbm')
#sim.counterfactual_Y    

#sim.optimization_result
#sim.optimize(smoothing = False)


#%% Evaluation

evaluation = utils.EvalMetrics(generator = generator, simulator = sim)
evaluation.graph_metrics()
#evaluation.graph_metrics
evaluation.error_counterfactuals([0,1], method = 'lgbm')
#evaluation.MSE_counterfactuals
#evaluation.MSE_mean_counterfactuals









