import pandas as pd
import numpy as np


graph_metrics = pd.read_pickle('eval_metrics/graph_metrics.pkl')
time_metrics = pd.read_pickle('eval_metrics/time_metrics.pkl')
mse_y = pd.read_pickle('eval_metrics/mse_y.pkl')
mse_mean = pd.read_pickle('eval_metrics/mse_mean.pkl')


res = {m: [] for m in list(graph_metrics.keys())}                                    
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


res_time = {}
for m in time_metrics.keys():
    res_time[m] = np.mean(list(map(lambda x: np.mean(x), mse_mean[m])))


print('graph metrics:', res)
print('MSE:', res_mean)
print('Mean:', res_mean)
print('time metrics:', res_time)

#ldecc seems to be the best one 



