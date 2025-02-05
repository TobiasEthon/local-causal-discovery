import pandas as pd
import numpy as np
import logging
from causal_discovery.utils import get_all_combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import networkx as nx
from castle.datasets import DAG
from scipy.special import expit as sigmoid
from sklearn.model_selection import GridSearchCV
import lightgbm

from causal_discovery.pc_alg import PCAlgorithm
from causal_discovery.mb_by_mb import MBbyMBAlgorithm
from causal_discovery.sd_alg import SequentialDiscoveryAlgorithm
from causal_discovery.ldecc import LDECCAlgorithm
from causal_discovery.mb_and_causal_discovery import MB_WITH_CD_ALGO
           
class Simulator(object):
    
    def __init__(self, df):
        self.df = df
    
    def estimate_parents(self, method = 'sd', cd_method = None):
        print('The parements are estimated using the causal discovery algorithm {}.'.format(method))
        if method == 'sd':
            algo = SequentialDiscoveryAlgorithm(use_ci_oracle=False)
            
        elif method == 'mb_by_mb':
            algo = MBbyMBAlgorithm(use_ci_oracle=False)
                
        elif method == 'ldecc':
            algo = LDECCAlgorithm(use_ci_oracle=False, ldecc_do_checks=True)
        
        elif method == 'MB_WITH_CD_ALGO':
            algo = MB_WITH_CD_ALGO(use_ci_oracle=False, cd_method=cd_method)
        
        else:
            raise ValueError('This method is currently not implemented. Use "sd", "mb_by_mb", "ldecc".')
                
        self.result = algo.run(self.df.copy())
        self.adjusted_result = self.result.copy()
        if 'Y' in self.adjusted_result['tmt_parents']:
            self.adjusted_result['tmt_parents'].remove('Y')
        
        if method in ['mb_by_mb', 'ldecc', 'sd']:
            self.parents_all = [list(self.result['tmt_parents']) + par for par in get_all_combinations(self.result["unoriented"], self.result["non_colliders"])]
            
            if 'Y' in self.adjusted_result['unoriented']:
                self.adjusted_result['unoriented'].remove('Y')
            
            combos = get_all_combinations(self.adjusted_result["unoriented"], self.adjusted_result["non_colliders"])
            self.parents = [list(self.adjusted_result['tmt_parents']) + par for par in combos]
        else:
            self.parents = [list(self.adjusted_result['tmt_parents'])]
            
            
    def parents_to_vector(self):
    
        label_to_idx = {j: i for i, j in enumerate(self.df.columns)}
        idx_parents = [[label_to_idx[l] for l in par] for par in self.parents]
        
        l_arr = []
        for idx_par in idx_parents:
            arr = np.zeros(self.df.shape[1])
            arr[idx_par] = 1
            l_arr.append(list(arr))
        
        self.parents_vector = l_arr
        
    def estimate_counterfactuals(self, X_intervention, method = 'lgbm', hyp_opt = False) -> list:
        print('The counterfactuals are estimated using {}.'.format(method)) 
        if method == 'lgbm': 
            if hyp_opt:
                hyp = {#'objective': metric,
                       'n_estimators': [5, 10, 50, 200],
                       'learning_rate': [0.001, 0.01, 0.1],
                       'min_data_in_leaf': [int(len(self.df)**.5)/2, int(len(self.df)**.5), 2*int(len(self.df)**.5)],
                       'num_leaves': 20 + int(min(max(np.log(len(self.df.columns) * 100), 0), 20)),
                       'deterministic': True,
                       'force_row_wise': True,  # This is recommended to use when deterministic=True with many data points.
                       }
            else:
                    
                n_estimators = 250
                #metric = 'regression_l1'
                hyp = {#'objective': metric,
                       'n_estimators': n_estimators,
                       'learning_rate': 10 / n_estimators,
                       'min_data_in_leaf': int(len(self.df)**.5),
                       'num_leaves': 20 + int(min(max(np.log(len(self.df.columns) * 100), 0), 20)),
                       'deterministic': True,
                       'force_row_wise': True,  # This is recommended to use when deterministic=True with many data points.
                       }
                
        
        #parents = [list(self.result['tmt_parents']) + par for par in get_all_combinations(self.result["unoriented"], self.result["non_colliders"])]
        
        l_models = []
        for par in self.parents:
            cov = ['X'] + par
            
            if method == 'lgbm':
                model = lightgbm.LGBMRegressor()
                
                if hyp_opt:
                    est = GridSearchCV(model, hyp)
                    est.fit(self.df[cov], self.df['Y'])
                    hyp = est.best_params_
                    
                model = lightgbm.LGBMRegressor(verbosity=-1, **hyp)

            elif method == 'linear':
                model = LinearRegression()
            
            elif method == 'nn':
                model = nn()

            else:
                raise ValueError('Method not implemented. Use "lgbm" or "linear".')
                                
            model.fit(self.df[cov], self.df['Y'])
            
            l_models.append(model)
            del model
        
        if isinstance(X_intervention, list):
            pass
        else:
            X_intervention = [X_intervention]
        
        counterfactual = []
        for i, par in enumerate(self.parents):
            cov = ['X'] + par
            model = l_models[i]
            
            if len(X_intervention) > 1:
                X_intervention_vals = np.random.uniform(low=X_intervention[0], high = X_intervention[1], size = 100)
            else:
                X_intervention_vals = X_intervention
            
            counterfactual_par = []
            for X_intervention_val in X_intervention_vals:
                df_intervention = self.df.copy()
                df_intervention['X'] = X_intervention_val
                
                #Make prediction for outcome when X set to X_intervention_val
                counterfactual_par.append(model.predict(df_intervention[cov]))
            
            counterfactual.append(list(sum(counterfactual_par)/len(counterfactual_par)))
            
        res = pd.DataFrame(counterfactual, columns = self.df.index.values, index = ['Y_parent_' + str(i) for i in range(len(counterfactual))]).transpose()        
            
        self.counterfactual_Y = res
        self.X_intervention = X_intervention
        self.fct_rel = l_models


    def optimize(self, smoothing = False):
        opt_range = [min(self.df['X']), max(self.df['X'])]
        x_val = np.linspace(opt_range[0], opt_range[1], num = 100)
        res, res_y = [], []
        
        for model, par in zip(self.fct_rel, self.parents):
            df_opt = self.df.copy()
            cov = ['X'] + par
            y = []
            
            for x in x_val:
                df_opt['X'] = x
                #predict y
                y.append(np.mean(model.predict(df_opt[cov])))
            
            y = pd.Series(y, index = x_val)
            
            if smoothing:
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess(y, x_val, frac = 2/(max(x_val)-min(x_val)))
                y = pd.Series(lowess[:, 1], index = x_val)
    
            idx_max = y.argmax()
            res_y.append(y)
            res.append((y.index[idx_max], y.iloc[idx_max]))
                        
        res_y = pd.DataFrame(res_y, columns = x_val, index = ['Y_parent_' + str(i) for i in range(len(res_y))]).transpose()        
        res_y.index.name = 'X_val'
            
        self.optimization_result = {'res': res_y, 'opt_X': res_y.idxmax(), 'opt_Y': res_y.max()}  
                

def get_parents_estimate(df: pd.DataFrame, estimate: dict) -> list:

    parents = [list(estimate['tmt_parents']) + par for par in get_all_combinations(estimate["unoriented"], estimate["non_colliders"])]
    label_to_idx = {j: i for i, j in enumerate(df.columns)}
    idx_parents = [[label_to_idx[l] for l in par] for par in parents]
    
    l_arr = []
    for idx_par in idx_parents:
        arr = np.zeros(df.shape[1])
        arr[idx_par] = 1
        l_arr.append(list(arr))
    
    return l_arr


def graph_metrics(parents_est, parents_true):
    
    precision, accuracy, f1, recall = [], [], [], []
    for par in parents_est:
        precision.append(precision_score(parents_true, par) if sum(par) != 0 else np.nan)
        accuracy.append(accuracy_score(parents_true, par))
        f1.append(f1_score(parents_true, par) if sum(par) != 0 & sum(parents_true) != 0 else np.nan)
        recall.append(recall_score(parents_true, par) if sum(parents_true) != 0 else np.nan)

    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}



def choose_idx_treatment(dag):
    #choose treatment idx
    idx_treatment = []
    G_nx =  nx.from_numpy_matrix(dag, create_using=nx.DiGraph)
    idx_target = dag.shape[0]-1
       
    ancestors = nx.ancestors(G_nx, idx_target)
    no_ancestor = list(set(G_nx.nodes) - ancestors - {dag.shape[0]-1})
    
    idx_treatment += list(np.random.choice(list(ancestors), size = min(2,len(ancestors)), replace = False))
    if len(no_ancestor) > 0:
        idx_treatment.append(np.random.choice(no_ancestor))
    return idx_treatment


class IIDGenerator(object):
    '''
    Simulate IID datasets for causal structure learning.
    Parameters
    ----------
    W: np.ndarray
        Weighted adjacency matrix for the target causal graph.
    n: int
        Number of samples for standard trainning dataset.
    method: str, (linear or nonlinear), default='linear'
        Distribution for standard trainning dataset.
    sem_type: str
        gauss, exp, gumbel, uniform, logistic (linear); 
        mlp, mim, gp, gp-add.
    noise_scale: float
        Scale parameter of noise distribution in linear SEM.
    '''

    def __init__(self):
        pass

    def generate_dag(self, n_nodes=20, degree=3, graph_level=5, weight_range=[0,5]):
    
        # generate true causal dag
        dag = DAG.hierarchical(n_nodes=n_nodes, degree=degree, graph_level=graph_level, weight_range=weight_range, seed=2)
        dag = dag.transpose()
        self.dag = dag
        self.B = (self.dag != 0).astype(int)
        self.dag_properties = {'n_nodes': n_nodes, 'degree': degree, 'graph_level': graph_level, 'weight_range': weight_range}
        logging.info('DAG generated with {} nodes, degree of {}, and graph level of {}'.format(n_nodes, degree, graph_level))
    
    def generate_data(self, n=2000, method='linear', sem_type='gauss', noise_scale=1.0):
        
        if method == 'linear':
            self.df, self.fct_rel = IIDGenerator._simulate_linear_sem(
                    self.dag, n, sem_type, noise_scale)
        elif method == 'nonlinear':
            self.df, self.fct_rel = IIDGenerator._simulate_nonlinear_sem(
                    self.dag, n, sem_type, noise_scale)
        
        #Name params and the target variable Y
        col_names = ['x_' + str(i) for i in range(self.df.shape[1]-1)] + ['Y']
        self.df = pd.DataFrame(self.df, columns = col_names)
        self.df_properties = {'n': n, 'method': method, 'sem_type': sem_type, 'noise_scale': noise_scale}   
        logging.info('Finished synthetic dataset')
        

    @staticmethod
    def _simulate_linear_sem(W, n, sem_type, noise_scale):
        """
        Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
        Parameters
        ----------
        W: np.ndarray
            [d, d] weighted adj matrix of DAG.
        n: int
            Number of samples, n=inf mimics population risk.
        sem_type: str 
            gauss, exp, gumbel, uniform, logistic.
        noise_scale: float 
            Scale parameter of noise distribution in linear SEM.
        
        Return
        ------
        X: np.ndarray
            [n, d] sample matrix, [d, d] if n=inf
        """
        def _simulate_single_equation(X, w, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""     
            if sem_type == 'logistic':
                fct_rel = logistic(w)
            else:
                fct_rel = linear(w)
     
            x = fct_rel.predict(X)
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x += z
            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x += z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x += z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x += z
            elif sem_type == 'logistic':
                x = np.random.binomial(1, x) * 1.0
            else:
                raise ValueError('Unknown sem type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, logistic.')
            return x, fct_rel

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale
        G_nx =  nx.from_numpy_matrix(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')
        #if np.isinf(n):  # population risk for linear gauss SEM
        #    if sem_type == 'gauss':
        #        # make 1/d X'X = true cov
        #        X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
        #        return X
        #    else:
        #        raise ValueError('population risk not available')
        # empirical risk
        
        #save functional relationships in dict
        f = {}
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            X[:, j], fct_rel_j = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
            f[str(j)] = fct_rel_j
        return X, f

    @staticmethod
    def _simulate_nonlinear_sem(W, n, sem_type, noise_scale):
        """
        Simulate samples from nonlinear SEM.
        Parameters
        ----------
        B: np.ndarray
            [d, d] binary adj matrix of DAG.
        n: int
            Number of samples.
        sem_type: str
            mlp, mim, gp, gp-add, or quadratic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.
        Return
        ------
        X: np.ndarray
            [n, d] sample matrix
        """

        def _simulate_single_equation(X, scale):
            """X: [n, num of parents], x: [n]"""
            z = np.random.normal(scale=scale, size=n)
            pa_size = X.shape[1]
            
            if pa_size == 0:
                return z, []
            if sem_type == 'mlp':
                fct_rel = mlp(pa_size)
                x = fct_rel.predict(X) + z
            elif sem_type == 'mim':
                fct_rel = mim(pa_size)
                x = fct_rel.predict(X) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                fct_rel = GaussianProcessRegressor()
                x = fct_rel.sample_y(X, random_state=None).flatten()
            elif sem_type == 'gp-add':
                fct_rel = gp_add()
                x = fct_rel.sample_y(X)
            else:
                raise ValueError('Unknown sem type. In a nonlinear model, the options are as follows: mlp, mim, gp, or gp-add.')
            return x, fct_rel

        B = (W != 0).astype(int)
        d = B.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        X = np.zeros([n, d])
        G_nx =  nx.from_numpy_matrix(B, create_using=nx.DiGraph)
        
        #save functional relationships in dict
        f = {}
        ordered_vertices = list(nx.topological_sort(G_nx))
        assert len(ordered_vertices) == d
        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            #print(_simulate_single_equation(X[:, parents], scale_vec[j]).shape)
            X[:, j], fct_rel_j = _simulate_single_equation(X[:, parents], scale_vec[j])
            f[str(j)] = fct_rel_j
        return X, f



    def define_treatment_variable(self, idx_treatment = None):
        self.reset_treatment_variable()
                
        if idx_treatment is None:
            #Choose randonly if nothing is given
            treatment = np.random.choice(self.df.columns.drop('Y'))
        else:
            treatment = self.df.columns[idx_treatment]
            
        self.df.rename(columns = {treatment: 'X'}, inplace = True)
        

    def reset_treatment_variable(self):
        # define treatment variable
        col_names = {self.df.columns[i]: 'x_' + str(i) for i in range(self.df.shape[1]-1)}
        self.df.rename(columns = col_names, inplace=True)

    def counterfactual_ground_truth(self, X_intervention):
        
        if isinstance(X_intervention, list):
            pass
        else:
            X_intervention = [X_intervention]

        self.df_intervention = IIDGenerator._predict_counterfactual(self, X_intervention)
        self.mean_outcome = self.df_intervention['Y'].mean()
        self.mean_diff = self.mean_outcome - self.df['Y'].mean()
        self.X_intervention = X_intervention


        logging.info('Finished counterfactual dataset')
        

    def _predict_counterfactual(self, X_intervention: list = []) -> pd.DataFrame:
        """
        Predicts from linear SEM when intervened.
        
        Parameters
        ----------
        X_intervention_val: np.float64
            
        
        Return
        ------
        df_updated: np.ndarray
            [n, d] updated observations
        """

        df_res = []

        if len(X_intervention) > 1:
            X_intervention_vals = np.random.uniform(low=X_intervention[0], high = X_intervention[1], size = 100)
        else:
            X_intervention_vals = X_intervention
            
        G_nx =  nx.from_numpy_matrix(self.dag, create_using=nx.DiGraph)
        idx_treatment = int(np.argwhere(self.df.columns == 'X'))
        
        for X_intervention_val in X_intervention_vals:
            df_updated = self.df.copy()
            df_updated['X'] = X_intervention_val
            children = list(G_nx.successors(idx_treatment))
        
            # empirical risk
            while children:
                j = children.pop(0)
                parents = list(G_nx.predecessors(j))
                df_updated.iloc[:, j] = self.fct_rel[str(j)].predict(df_updated.iloc[:, parents].values)
                children += list(G_nx.successors(j))
            df_res.append(df_updated)
            del df_updated
            
        return sum(df_res)/len(df_res)

    def parents_to_vector(self):
        label_to_idx = {j: i for i, j in enumerate(self.df.columns)}
        idx_treatment = label_to_idx['X']
        
        self.parents_vector = list((self.dag[:, idx_treatment] != 0).astype(int))


    def optimize(self):
        opt_range = [min(self.df['X']), max(self.df['X'])]
        x_val = np.linspace(opt_range[0], opt_range[1], num = 100)
        res_y = []
        
        for x in x_val:
            pred = self._predict_counterfactual([x])
            res_y.append(np.mean(pred['Y']))
           
        res = pd.Series(res_y, index = x_val, name = 'Y')
        res.index.name = 'X_val'

        self.optimization_result = {'res': res, 'opt_X': res.idxmax(), 'opt_Y': res.max()}  
        


class EvalMetrics(object):
    
    def __init__(self, generator, simulator):
        self.generator = generator
        self.simulator = simulator

        self.generator.parents_to_vector()
        self.simulator.parents_to_vector()
        
        
    def graph_metrics(self):
        from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
        precision, accuracy, f1, recall = [], [], [], []
        for par in self.simulator.parents_vector:
            precision.append(precision_score(self.generator.parents_vector, par) if sum(par) != 0 else np.nan)
            accuracy.append(accuracy_score(self.generator.parents_vector, par))
            f1.append(f1_score(self.generator.parents_vector, par) if sum(par) != 0 & sum(self.generator.parents_vector) != 0 else np.nan)
            recall.append(recall_score(self.generator.parents_vector, par) if sum(self.generator.parents_vector) != 0 else np.nan)

        self.graph_metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
        
        
    def error_counterfactuals(self, X_intervention, method = 'lgbm'):
        from sklearn.metrics import mean_squared_error
        
        #if not hasattr(self.simulator, 'X_intervention') or self.simulator.X_intervention != X_intervention:
        self.simulator.estimate_counterfactuals(X_intervention, method)
        
        #if not hasattr(self.generator, 'X_intervention') or self.generator.X_intervention != X_intervention:
        self.generator.counterfactual_ground_truth(X_intervention)       


        self.MSE_counterfactuals = [mean_squared_error(self.generator.df_intervention['Y'].values, self.simulator.counterfactual_Y[col])
                                    for col in self.simulator.counterfactual_Y.columns]
        
        self.MSE_mean_counterfactuals = [(np.mean(self.generator.df_intervention['Y'].values) - np.mean(self.simulator.counterfactual_Y[col]))**2
                                         for col in self.simulator.counterfactual_Y.columns]    
        



class linear(object):
    
    def __init__(self, w):
        self.coef = w
        
    def predict(self, X):
        return X @ self.coef

class logistic(object):
    
    def __init__(self, w):
        self.coef = w
        
    def predict(self, X):
        return sigmoid(X @ self.coef)


class mlp(object):
    
    def __init__(self, pa_size):
        self.pa_size = pa_size
        self.hidden = 100
        W1 = np.random.uniform(low=0.5, high=2.0, size=[self.pa_size, self.hidden])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        self.W1 = W1
        W2 = np.random.uniform(low=0.5, high=2.0, size=self.hidden)
        W2[np.random.rand(self.hidden) < 0.5] *= -1
        self.W2 = W2
        
    def predict(self, X):
        return sigmoid(X @ self.W1) @ self.W2



class mim(object):
    
    def __init__(self, pa_size):
        self.pa_size = pa_size
        w1 = np.random.uniform(low=0.5, high=2.0, size=self.pa_size)
        w1[np.random.rand(self.pa_size) < 0.5] *= -1
        self.w1 = w1
        w2 = np.random.uniform(low=0.5, high=2.0, size=self.pa_size)
        w2[np.random.rand(self.pa_size) < 0.5] *= -1
        self.w2 = w2
        w3 = np.random.uniform(low=0.5, high=2.0, size=self.pa_size)
        w3[np.random.rand(self.pa_size) < 0.5] *= -1
        self.w3 = w3
        
    def predict(self, X):   
       return np.tanh(X @ self.w1) + np.cos(X @ self.w2) + np.sin(X @ self.w3)

class gp_add(object):
    
    def __init__(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        self.gp = GaussianProcessRegressor()
                
    def sample_y(self, X):
        return sum([self.gp.sample_y(X[:, i, None]).flatten() for i in range(X.shape[1])])

    def predict(self, X):
        return sum([self.gp.predict(X[:, i, None]).flatten() for i in range(X.shape[1])])



