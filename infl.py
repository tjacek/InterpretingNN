import numpy as np
from tqdm import tqdm
import base,clf,dataset,utils

def infl_full( in_path,
               split_path,
               out_path=None):    
    split_dict=base.get_split_dict(split_path)
    @utils.dir_fun
    def helper(in_path,out_path=None):
        data_i=dataset.read_csv(in_path)
        id_i=in_path.split("/")[-1]
        print(id_i)
        split_i=split_dict[id_i]
        infl=eval_split(split_i,data_i)
        print(infl)
    helper(in_path,out_path)

def infl_binary( in_path,
               split_path,
               out_path=None):    
    split_dict=base.get_split_dict(split_path)
    @utils.dir_fun
    def helper(in_path,out_path=None):
        data_i=dataset.read_csv(in_path)
        id_i=in_path.split("/")[-1]
        print(id_i)
        split_i=split_dict[id_i]
        infl_matrix=[]
        for j in range(data_i.n_cats()):
            data_j=data_i.binarize(j)
            infl_j=eval_split(split_i,data_j)
            infl_matrix.append(infl_j)
        infl_matrix=np.array(infl_matrix)
        print(infl_matrix)
        np.savetxt(out_path, infl_matrix, fmt='%f')
    helper(in_path,out_path)

def eval_split(split_i,data_i):
    result_i,_=split_i(data_i,clf.RF)
    f1=result_i.get_f1()
    infl=[]
    for j in tqdm(range(data_i.dim())):
          data_j=data_i.remove_col(j)
          result_j,_=split_i(data_j,clf.RF,verbose=False)
          infl.append(f1-result_j.get_f1())
    return np.array(infl)

#infl_binary("uci/ineq",
#        "uci/output",
#	    "test/f1")
import shape
shape.show_heatmap("test/f1")
