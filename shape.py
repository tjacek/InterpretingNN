import numpy as np
import shap
import argparse
import os,os.path
from tqdm import tqdm
import base
import plot
import dataset
import dist
import exp
import utils

class ShapleyGroup(dict):
    def __iter__(self):
        for value_i in self.values():
            yield value_i
    
    def select(self,text):
        s_dict={key_i:value_i
                   for key_i,value_i in self.items()
                    if(text in key_i)}
        return ShapleyGroup(s_dict)

    def by_id(self,data_id):
        for key_i,value_i in self.items():
            if( data_id in key_i):
                return value_i 

    @classmethod 
    def read(cls,in_path):
        paths=utils.bottom_files(in_path)
        shap_matrices={}
        for path_i in paths:
            id_i=path_i.replace(in_path,"")
            id_i=id_i.replace("/","-")
            shap_i=ShapleyMatrix( id=id_i,
                                  matrix=read(path_i))
            shap_matrices[id_i]=shap_i
        return cls(shap_matrices)

class ShapleyMatrix(object):
    def __init__( self,
                  id,
                  matrix):
        self.id=id 
        self.matrix=matrix
    
    def density(self):
        points=self.matrix.flatten()
        points-=np.mean(points)
        points/=np.std(points)
        return dist.EmpDist.make(points)

    def __str__(self):
        return self.id
    
    def __repr__(self):
        return self.id

def read(in_path):
    all_shap=[]
    for id_i, path_j in utils.iter_files(in_path):
        shap_j=np.load(path_j)["arr_0"]
        all_shap.append(shap_j)
    shap_arr=np.concatenate(all_shap,axis=0)
    return np.mean(shap_arr,axis=0)

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
    utils.make_dir(shap_exp.out_path)
    for i,split_i in enumerate(tqdm(splits)):
        out_i=f"{shap_exp.out_path}/{i}"
        clf_i,_=split_i.fit_clf(data,clf_type())
        print(f"Clf trained:{clf_i}")
        values_i=helper(split_i,clf_i)
        np.savez(out_i, values_i)

def compute_shapley(conf_path):
    conf=utils.read_json(conf_path)
    exps=exp.from_conf(conf)
    for exp_i in tqdm(exps):
        make_shap(exp_i)

def get_shapley(conf_path):
    conf=utils.read_json(conf_path)
    out_path=conf["out_path"]
    return ShapleyGroup.read(out_path)

def show_shapley(conf_path):
    shap_matrices=get_shapley(conf_path)
    for shap_i in shap_matrices:
        plot.show_heatmap( shap_i.matrix,
                           shap_i.id)

def dist_shapley(conf_path):
    shap_matrices=get_shapley(conf_path)
    for shap_i in shap_matrices:
        dist_i=shap_i.density()
        dist_i.plot(shap_i.id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="uci/conf.json") 
    parser.add_argument("--cmd", type=str,default="dist")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        compute_shapley(args.conf_path)
    if(args.cmd=="show"):
        show_shapley(  args.conf_path)
    if(args.cmd=="dist"):
        dist_shapley(  args.conf_path)