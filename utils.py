import pandas as pd
import numpy as np
from causal_discovery.utils import get_all_combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import networkx as nx
from castle.datasets import IIDSimulation, DAG
from sklearn.metrics import confusion_matrix


def get_ATE(estimate: dict, df: pd.DataFrame, X_intervention_val: np.float64 = None, 
                        X_intervention_interval: list = []) -> list:
    
    n_estimators = 250
    lightgbm_parameters = {'objective': 'regression_l2',
                           'n_estimators': n_estimators,
                           'learning_rate': 10 / n_estimators,
                           'num_leaves': 20,
                           'feature_fraction_bynode': 0.25,
                           }
    
    
    parents = [list(estimate['tmt_parents']) + par for par in get_all_combinations(estimate["unoriented"], estimate["non_colliders"])]
    
    target = ['Y']
    l_models = []
    for par in parents:
        cov = ['X'] + par
        
        #model = lightgbm.LGBMRegressor(**lightgbm_parameters)
        model = LinearRegression()
        model.fit(df[cov], df[target])
        
        l_models.append(model)
        del model
    
    ATE = []
    for i, par in enumerate(parents):
        cov = ['X'] + par
        model = l_models[i]
        
        if X_intervention_interval:
            X_intervention_vals = np.random.uniform(low=X_intervention_interval[0], high = X_intervention_interval[1], size = 2)
        else:
            X_intervention_vals = [X_intervention_val]
        
        for X_intervention_val in X_intervention_vals:
            ate_par = []
            df_intervention = df.copy()
            df_intervention['X'] = X_intervention_val
            
            #Make prediction for outcome when X set to X_intervention_val
            ate_par.append(model.predict(df_intervention[cov]))
        ATE.append(np.mean(ate_par))
        
    return ATE


def _predict_linear_sem(W: np.array, df: pd.DataFrame, X_intervention_val: np.float64 = None, 
                        X_intervention_interval: list = []) -> pd.DataFrame:
    """
    Predicts from linear SEM when intervened.
    
    Parameters
    ----------
    W: np.ndarray
        [d, d] weighted adj matrix of DAG.
    df: pd.DataFrame
        [n, d] observations
    X_intervention_val: np.float64
        
    
    Return
    ------
    df_updated: np.ndarray
        [n, d] updated observations
    """
    def _predict_single_equation(X: pd.DataFrame, w: np.array) -> pd.DataFrame:
        """X: [n, num of parents], w: [num of parents], x: [n]"""

        return X @ w

    df_res = []

    if X_intervention_interval:
        X_intervention_vals = np.random.uniform(low=X_intervention_interval[0], high = X_intervention_interval[1], size = 2)
    else:
        X_intervention_vals = [X_intervention_val]
        
    G_nx =  nx.from_numpy_matrix(W, create_using=nx.DiGraph)
    idx_treatment = int(np.argwhere(df.columns == 'X'))
    
    for X_intervention_val in X_intervention_vals:
        df_updated = df.copy()
        df_updated['X'] = X_intervention_val
        children = list(G_nx.successors(idx_treatment))
    
        # empirical risk
        while children:
            j = children.pop(0)
            parents = list(G_nx.predecessors(j))
            df_updated.iloc[:, j] = _predict_single_equation(df_updated.iloc[:, parents].values, W[parents, j])
            children += list(G_nx.successors(j))
        df_res.append(df_updated)
        del df_updated
        
    return sum(df_res)/len(df_res)


def get_parents_true(df: pd.DataFrame, graph: np.ndarray) -> np.ndarray:
    label_to_idx = {j: i for i, j in enumerate(df.columns)}
    idx_treatment = label_to_idx['X']
    return list((graph[:, idx_treatment] != 0).astype(int))


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


def generate_graph(n_nodes=20, degree=3, graph_level=5, weight_range=[0,5]):

    # generate true causal dag
    dag = DAG.hierarchical(n_nodes=n_nodes, degree=degree, graph_level=graph_level, weight_range=weight_range ,seed=1)
    dag = dag.transpose()
    return dag

def generate_data(dag, idx_treatment, n=2000, method='linear', sem_type='gauss', noise_scale=1.0):
    dataset = IIDSimulation(W=dag, n=n, method=method, sem_type=sem_type, noise_scale = noise_scale)

    #Name params and the target variable Y
    col_names = ['x_' + str(i) for i in range(dataset.X.shape[1]-1)] + ['Y']
    df = pd.DataFrame(dataset.X, columns = col_names)

    return df


def define_treatment_variable(df):
    
    # define treatment variable
    col_names = {df.columns[i]: 'x_' + str(i) for i in range(df.shape[1]-1)}
    df.rename(columns = col_names, inplace=True)

    # define treatment variable
    treatment = np.random.choice(df.columns.drop('Y'))
    df.rename(columns = {treatment: 'X'}, inplace = True)

    return df


