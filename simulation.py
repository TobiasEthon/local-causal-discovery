import numpy as np
import pandas as pd

from causal_discovery.utils import *
from causal_discovery.pc_alg import PCAlgorithm
from causal_discovery.mb_by_mb import MBbyMBAlgorithm
from causal_discovery.sd_alg import SequentialDiscoveryAlgorithm
from causal_discovery.ldecc import LDECCAlgorithm

from lingam.utils import make_dot

import utils


#%%

n_nodes = 10
degree=3
graph_level=5
weight_range=[0, 5]

dag = utils.generate_graph(n_nodes, degree, graph_level, weight_range)

#Generate data
method = 'linear'
sem_type = 'gauss'
n = 2000
idx_treatment = 5
noise_scale = 1.0

df = utils.generate_data(dag, idx_treatment, n, method, sem_type, noise_scale)

#make_dot(dag.transpose(), labels = list(df.columns))


#%% Run algos

df = utils.define_treatment_variable(df)

# algos
mb_by_mb_alg = MBbyMBAlgorithm(use_ci_oracle=False)
result_mb_by_mb = mb_by_mb_alg.run(df)

sd_alg = SequentialDiscoveryAlgorithm(use_ci_oracle=False)
result_sd = sd_alg.run(df)

ldecc_alg = LDECCAlgorithm(use_ci_oracle=False, ldecc_do_checks=True)
result_ldecc = ldecc_alg.run(df)


#%% Evaluate the quality of the subgraph (multi-classification)

parents_ldecc = utils.get_parents_estimate(df, result_ldecc)
parents_sd = utils.get_parents_estimate(df, result_sd)
parents_mb_by_mb = utils.get_parents_estimate(df, result_mb_by_mb)

true_parents = utils.get_parents_true(df, dag)

utils.graph_metrics(parents_ldecc, true_parents)
utils.graph_metrics(parents_sd, true_parents)
utils.graph_metrics(parents_mb_by_mb, true_parents)


#%% Calculate ATE for each possible graph/set of parents

#Here, I could use what Marco did

X_intervention_val = df['X'].quantile(0.75)
ate_ldecc = utils.get_ATE(result_ldecc, df, X_intervention_interval=[0,1])
ate_sd = utils.get_ATE(result_sd, df, X_intervention_val)
ate_mb_by_mb = utils.get_ATE(result_mb_by_mb, df, X_intervention_val)

print(ate_ldecc)
print(ate_sd)
print(ate_mb_by_mb)


#%% Get ground truth ATE

outcome_true = utils._predict_linear_sem(dag, df, X_intervention_interval=[-10,10])
print(outcome_true['Y'].mean())


#ground truths for all DGPs










