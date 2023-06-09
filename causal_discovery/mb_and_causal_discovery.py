import numpy as np
import pandas as pd
import networkx as nx
import numpy
import pandas
import networkx
from itertools import combinations, permutations
import logging
from causallearn.utils.cit import CIT
from sklearn import linear_model
import collections
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
import copy
import networkx as nx


from causal_discovery.utils import *
from causal_discovery.causal_discovery_base import CausalDiscoveryBase


from castle.algorithms import PC
from castle.algorithms import GOLEM
from castle.algorithms import Notears
from castle.algorithms.ges.ges import GES
from castle.algorithms import NotearsNonlinear
from castle.algorithms import ANMNonlinear
from castle.common.priori_knowledge import PrioriKnowledge


class MB_WITH_CD_ALGO(CausalDiscoveryBase):

    def __init__(self, treatment_node="X", outcome_node="Y", alpha=0.05, 
                 use_ci_oracle=True, graph_true=None, enable_logging=False,
                 ldecc_do_checks=False, max_tests=None, cd_method='pc'):
        
        super(MB_WITH_CD_ALGO, self).__init__(treatment_node=treatment_node,
                                          outcome_node=outcome_node, alpha=alpha,
                                          use_ci_oracle=use_ci_oracle,
                                          graph_true=graph_true,
                                          enable_logging=enable_logging,
                                          max_tests=max_tests)
        self.cd_method = cd_method
    
    def run(self, data):
    
        cols = data.columns
        self.cols = cols
        return self._run_mb_no_tears(self._partial_corr_independence_test,
                               pd.DataFrame(data=data.values, columns=range(data.values.shape[1])),
                               alpha=self.alpha, treatment_node_id=list(cols).index(self.tmt_node),
                               outcome_node_id=list(cols).index(self.outcome_node), cd_method=self.cd_method)
          

    def _run_mb_no_tears(self, indep_test_func, data_matrix, alpha, treatment_node_id,
                   outcome_node_id, cd_method):
    
            
        def is_independent(i, j, k, ret_p_val=False):
          p_val = indep_test_func(data_matrix, i, j, k)
          is_indep = p_val > alpha
        
          if ret_p_val:
            return is_indep, p_val
          else:
            return is_indep
    
        def find_markov_blanket(g, sep_set, node_ids, main_node_id):
          g = g.copy()
          sep_set = copy.deepcopy(sep_set)
          markov_blanket = set()
    
          # forward pass.
          cont = True
          while cont:
            cont = False
            mb_copy = set(markov_blanket)
            nodes_to_check = (node_ids - {main_node_id} - mb_copy)
            for n in nodes_to_check:
              if not is_independent(n, main_node_id, markov_blanket - {n}):
                markov_blanket |= {n}
                cont = True
              elif g.has_edge(n, main_node_id):
                g.remove_edge(n, main_node_id)
                sep_set[n][main_node_id] = markov_blanket - {n}
                sep_set[main_node_id][n] = markov_blanket - {n}
            
          # backward pass.
          mb_copy = set(markov_blanket)
          for n in mb_copy:
            if is_independent(n, main_node_id, markov_blanket - {n}):
              if g.has_edge(n, main_node_id):
                g.remove_edge(n, main_node_id)
                sep_set[n][main_node_id] = markov_blanket - {n}
                sep_set[main_node_id][n] = markov_blanket - {n}
              
              markov_blanket -= {n}
          
          return g, sep_set, markov_blanket

        tmt = treatment_node_id
        outcome = outcome_node_id
        self.tmt_id = tmt
                
        node_ids = set(range(data_matrix.shape[1]))
        node_size = data_matrix.shape[1]
        sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
        g = self._create_complete_graph(node_ids)

        # We begin by finding neighbors Ne(X). For this, we first find the Markov 
        # blanket MB(X) and then run the PC algorithm locally within MB(X) to find
        # Ne(X).

        g, sep_set, markov_blanket = (
            find_markov_blanket(g, sep_set, node_ids, treatment_node_id)
        )

        #Run the causal discovery algorithm (PC for now) on the markov blanket
        data_matrix_markov_blanket = data_matrix.iloc[:, list(markov_blanket) + [tmt]]
        
        priori = PrioriKnowledge(data_matrix_markov_blanket.shape[1])
        #Add domain knowledge
        
        col_markov_blanket = list(data_matrix_markov_blanket.columns)        

        if cd_method == 'pc':            
            cd = PC()
            cd.learn(data_matrix_markov_blanket)
                    
        elif cd_method == 'golem':            
            # GOLEM learn
            cd = GOLEM(num_iter=1e4)
            cd.learn(data_matrix_markov_blanket)
            
        elif cd_method == 'no_tears':
            # notears learn
            cd = Notears()
            cd.learn(data_matrix_markov_blanket)
            
        elif cd_method == 'ges':
            cd = GES(criterion='bic', method='scatter')
            cd.learn(data_matrix_markov_blanket)
            
        elif cd_method == 'non_linear_no_tears':
            # notears-nonlinear learn
            cd = NotearsNonlinear()
            cd.learn(data_matrix_markov_blanket)
            
        elif cd_method == 'anm':
            cd = ANMNonlinear(alpha=0.05)
            cd.learn(data=data_matrix_markov_blanket)
            
        causal_matrix = pd.DataFrame(cd.causal_matrix, index = col_markov_blanket, columns = col_markov_blanket)
        
        
        
        
        tmt_parents = set(data_matrix_markov_blanket.columns[causal_matrix.loc[:, tmt].astype(bool)])
        tmt_children = set(data_matrix_markov_blanket.columns[causal_matrix.loc[tmt, :].astype(bool)])            
   
        tmt_parents -= tmt_children
        
        self.log("Parents: %s" % str(set(self.cols[o] for o in tmt_parents)))
        self.log("Children: %s" % str(set(self.cols[o] for o in tmt_children)))
        
        return {
            "tmt_parents": set(self.cols[o] for o in tmt_parents),
            "tmt_children": set(self.cols[o] for o in tmt_children)}
         
    