import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse
import xgboost
import base,dataset,plot,utils

def rf_shapley( in_path,
                split_path,
                out_path):
    split_dict=base.get_split_dict(split_path)
    @utils.dir_fun
    def helper(in_path,out_path=None):
        data_id=in_path.split("/")[-1]
        print(data_id)
        data=dataset.read_csv(in_path)
        splits=split_dict[data_id]
        all_shape=[]
        for split_i in tqdm(splits.splits):
            shap_i=split_shapley(split_i,data)
            all_shape.append(shap_i)
        shap_arr=np.concatenate(all_shape,axis=0)
        shap_matrix=np.mean(shap_arr,axis=0)
        np.savetxt(out_path, shap_matrix, fmt='%f')
    helper(in_path,out_path)

def split_shapley(split,data):
    train,test=split(data)
    clf=RandomForestClassifier(class_weight="balanced")
    clf.fit(train.X,train.y)
    explainer = shap.Explainer(clf)
    shap_values = explainer(test.X)
    return shap_values.values

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,default="uci/ineq")
    parser.add_argument("--split", type=str,default="uci/output")
    parser.add_argument("--output", type=str,default="test/shapley")
    args=parser.parse_args()
    rf_shapley( in_path=args.input,
                split_path=args.split,
                out_path=args.output)
#    show_heatmap(args.heat)