import numpy as np
import pandas as pd
import utils

from tensorflow import keras





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


df = generator.df







