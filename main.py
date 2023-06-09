import numpy as np
import pandas as pd
import pickle 

from lingam.utils import make_dot

import utils
import time



n_nodes = [10, 20, 30, 50, 100]
degree = [3, 8]
#graph_level=[5, 10, 15]
weight_range=[[0, 5]] #I think only relevent for linear methods, Should I use None and then creat lin reg weights randomly ?

method = ['linear', 'nonlinear']
sem_type = {'linear': ['gauss', 'exp'], 'nonlinear': ['mlp']}

N = [100, 1000, 10000, 100000]
noise_scale = [0.1, 1, 10]

method_sim = ['ldecc', 'MB_WITH_CD_ALGO']#, 'mb_by_mb'
method_cd = ['pc', 'no_tears', 'golem', 'non_linear_no_tears', 'anm']
method_fct = ['lgbm']

graph_metrics = {m: [] for m in method_sim}
graph_metrics.pop('MB_WITH_CD_ALGO')
[graph_metrics.update({m: []}) for m in method_cd]

mse_y = {m: [] for m in method_sim}
mse_y.pop('MB_WITH_CD_ALGO')
[mse_y.update({m: []}) for m in method_cd]

mse_mean = {m: [] for m in method_sim}
mse_mean.pop('MB_WITH_CD_ALGO')
[mse_mean.update({m: []}) for m in method_cd]

time_metrics = {m: [] for m in method_sim}
time_metrics.pop('MB_WITH_CD_ALGO')
[time_metrics.update({m: []}) for m in method_cd]



for n_node in n_nodes:
    for deg in degree:
        for graph_l in [int(n_node), int(n_node/2), int(n_node/4)]:
            for weight_r in weight_range:
                #Setup the generator
                generator = utils.IIDGenerator()
                generator.generate_dag(n_node, deg, graph_l, weight_r)

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
                                            if met_sim == 'MB_WITH_CD_ALGO':
                                                for met_cd in method_cd:
                                                    #Setup the simulator
                                                    t0 = time.time()
                                                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                                                    print('Method start running:', met_cd)
                                                    print('Start time:', t0)
                                                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                                                    sim = utils.Simulator(generator.df.copy())
                                                    sim.estimate_parents(method = met_sim, cd_method = met_cd)
                                                    X_int = sim.df.iloc[:, idx_t].quantile(0.66)
                                                    t1 = time.time()
                                                    
                                                    #Setup Evaluation
                                                    evaluation = utils.EvalMetrics(generator = generator, simulator = sim)
                                                    evaluation.graph_metrics()
                                                    graph_metrics[met_cd].append(evaluation.graph_metrics)
                                                    evaluation.error_counterfactuals(X_int, method = met_fct)
                                                    mse_y[met_cd].append(evaluation.MSE_counterfactuals)
                                                    mse_mean[met_cd].append(evaluation.MSE_mean_counterfactuals)
                                                    time_metrics[met_cd].append(t1-t0)
                                                    print('This took:', t1-t0)
                                                    
                                                    del sim
                                                    del evaluation
                                            else:
                                                t0 = time.time()
                                                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                                                print('Method start running:', met_sim)
                                                print('Start time:', t0)
                                                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                                                #Setup the simulator
                                                sim = utils.Simulator(generator.df.copy())
                                                sim.estimate_parents(method = met_sim)
                                                X_int = sim.df.iloc[:, idx_t].quantile(0.66)
                                                t1 = time.time()
                                                
                                                #Setup Evaluation
                                                evaluation = utils.EvalMetrics(generator = generator, simulator = sim)
                                                evaluation.graph_metrics()
                                                graph_metrics[met_sim].append(evaluation.graph_metrics)
                                                evaluation.error_counterfactuals(X_int, method = met_fct)
                                                mse_y[met_sim].append(evaluation.MSE_counterfactuals)
                                                mse_mean[met_sim].append(evaluation.MSE_mean_counterfactuals)
                                                time_metrics[met_sim].append(t1-t0)
                                                time_metrics[met_sim].append(t1-t0)
                                                
                                                del sim
                                                del evaluation



with open('eval_metrics/graph_metrics.pkl', 'wb') as f:
    pickle.dump(graph_metrics, f)

with open('eval_metrics/mse_y.pkl', 'wb') as f:
    pickle.dump(mse_y, f)

with open('eval_metrics/mse_mean.pkl', 'wb') as f:
    pickle.dump(mse_mean, f)

with open('eval_metrics/time_metrics.pkl', 'wb') as f:
    pickle.dump(time_metrics, f)

