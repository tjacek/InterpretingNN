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
        infl=eval_split(split_i,data_i)
        print(infl)
    helper(in_path,out_path)

#        split_i= base.Split.random(data_i)
#        acc=eval_split(split_i,data_i)
#        infl_i=[]
#        for j in tqdm(range(data_i.dim())):
#    	    data_j=data_i.remove_col(j)
#    	    acc_j=eval_split(split_i,data_i)
#    	    infl_i.append(acc-acc_j)
#        print(infl_i)

def eval_split(split_i,data_i):
    result_i,_=split_i(data_i,clf.RF)
    acc=result_i.get_acc()
    infl=[]
    for j in tqdm(range(data_i.dim())):
          data_j=data_i.remove_col(j)
          result_j,_=split_i(data_j,clf.RF,verbose=False)
          infl.append(acc-result_j.get_acc())
    return np.array(infl)

infl_full("uci/ineq",
        "uci/output",
	    "test")