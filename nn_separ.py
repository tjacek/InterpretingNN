import nn_dist


def cosine_matrix(in_path):
    for id_i,mlp_i in nn_dist.mlp_iter(in_path):
        print(id_i)
        print(mlp_i)


cosine_matrix("uci/output")