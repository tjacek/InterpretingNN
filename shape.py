import numpy as np
import shap
import argparse
import os,os.path
from tqdm import tqdm
import base
import plot
import dataset
import exp
import utils

def make_shap(shap_exp):
    data,splits=shap_exp.get_data()
    clf_type=shap_exp.get_clf()
    def helper(split_i,clf_i):
        train,test=data.divide(split_i)
        if((shap_exp.k is None) or 
            (shap_exp.k > test.X.shape[0])):
            background_data=train.X
            test_data=test.X
        else:
            background_data = shap.kmeans( train.X, 
                                           shap_exp.k).data
            test_data =shap.kmeans( test.X, 
                                    shap_exp.k).data
        explainer=shap.Explainer( clf_i.model.predict_proba,
                                  background_data)
        shap_values = explainer(test_data)
        return shap_values.values
    print(shap_exp.out_path)
    base.make_dir(shap_exp.out_path)
    for i,split_i in enumerate(tqdm(splits)):
        out_i=f"{shap_exp.out_path}/{i}"
        clf_i,_=split_i.fit_clf(data,clf_type())
        print(f"Clf trained:{clf_i}")
        values_i=helper(split_i,clf_i)
        np.savez(out_i, values_i)

def compute_shapley(conf_path):
    conf=utils.read_json(conf_path)
    exp.from_conf(conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="uci/conf.json") 
    parser.add_argument("--regex", type=str,default=r"MLP_(.)+")
    parser.add_argument("--cmd", type=str,default="compute")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        compute_shapley(args.conf_path)
#    if(args.cmd=="show"):
#        show_shapley(  args.conf_path,
#                       regex=args.regex)
#    if(args.cmd=="var"):
#        var_matrix( args.conf_path,
#                    regex=args.regex)