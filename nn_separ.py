import numpy as np
from itertools import combinations
from scipy.spatial.distance import cosine
import nn_dist,utils

def neural_spar(in_path,out_path=None):
    if(out_path):
        utils.make_dir(out_path)
    for id_i,mlp_i in nn_dist.mlp_iter(in_path):
        print(id_i)
        weights_i=mlp_i.get_weights(-1)
        matrix=cosine_matrix(weights_i)
        if(out_path):
            out_i=f'{out_path}/{id_i}.txt'
            np.savetxt(out_i, matrix, fmt='%f')

def cosine_matrix(weights):
    w=[w_i for w_i in weights.T]
    n_cats=len(w)
    matrix=np.zeros((n_cats,n_cats))
    for x, y in combinations(range(n_cats), 2):
        dist_xy=cosine(w[x],w[y])
        matrix[x][y]=dist_xy
        matrix[y][x]=dist_xy
    return matrix


neural_spar("uci/output","uci/cosine")