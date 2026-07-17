import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import argparse
import xgboost
import base,dataset,plot,utils

def compute_shapley(in_path,out_path=None):
    if(out_path):
        utils.make_dir(out_path)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        split_i=base.Split.random(data_i,0.9)
        clf_i=RandomForestClassifier()
        split_i.fit_clf(data_i,clf_i)
        train,test=data_i.divide(split_i)

        explainer = shap.Explainer(clf_i.predict_proba, 
         	                      train.X,
                                  algorithm="tree")
        shap_values = explainer(test.X,max_evals=620)
        values=np.mean(np.abs(shap_values.values),axis=0)
        if(out_path):
            id_i=path_i.split("/")[-1]
            np.savetxt(f'{out_path}/{id_i}.txt', values, fmt='%f')

def xboost_basic(in_path,out_path=None):
    if(out_path):
        utils.make_dir(out_path)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        
        model = xgboost.XGBRegressor().fit(data_i.X,data_i.y)
        explainer = shap.Explainer(model)
        shap_values = explainer(data_i.X)
        print(path_i)
        print(shap_values.shape)
        print(shap_values[0].shape)
#        shap.plots.waterfall(shap_values[0])
        shap.plots.beeswarm(shap_values)

def xboost_shapley(in_path,out_path=None):
    if(out_path):
        utils.make_dir(out_path)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        
        model = xgboost.XGBRegressor().fit(data_i.X,data_i.y)
        explainer = shap.Explainer(model)
        shap_values = explainer(data_i.X)
        print(path_i)

#def show_prop(in_path):
#    def helper(id,path):
#        values = np.loadtxt(path)
#        np.arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute', action='store_true')
    parser.add_argument("--heat", type=str,default="shapley")
    args=parser.parse_args()
    if(args.compute):
#        raise Exception("Wrong")
        xboost_shapley([#"AutoML/data",
                        "uci/data"],"shapley/xboost")	
#    show_heatmap(args.heat)