import numpy as np
import shap
import argparse
import clfs
import dataset
import main

def compute_shapley(  data_path,
	                  dir_path,
	                  clf="TabPNF"):
    dir_proxy=main.DirProxy(dir_path,clf)
    splits=dir_proxy.get_splits()
    clf_type=clfs.TYPES[clf]
    data=dataset.read_csv(data_path)
    results,models=splits(data,clf_type)
    shapley_path=dir_proxy.dispatch("shapley")
#    all_shape=[]
    for i,(split_i,model_i) in enumerate(zip(splits.splits,models)):
        values_i=shape_split(data,split_i,model_i)
        np.savez(f"{shapley_path}/{i}", values_i)
#        all_shape.append(values_i)
#    shap_arr=np.concatenate(all_shape,axis=0)
#    shap_matrix=np.mean(shap_arr,axis=0)
#    np.savetxt(out_path, shap_matrix, fmt='%f')

def shape_split(data_i,split_i,model_i):
    train,test=data_i.divide(split_i)

    explainer = shap.Explainer(model_i.proba_fun(), 
         	                   train.X)#,
#                                  algorithm="tree")
    shap_values = explainer(test.X,max_evals=620)
    return shap_values.values

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,default="selected/data/cleveland")
    parser.add_argument("--dir_path", type=str,default="selected/output/cleveland")
    parser.add_argument("--clf", type=str,default="TabPNF")

    args=parser.parse_args()
    compute_shapley( args.data_path,
	                 args.dir_path,
	                 args.clf)