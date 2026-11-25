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
    
    def datasets(self):
        raw=[]
        for shap_i in self.values():
            raw.append(shap_i.data)
        return list(set(raw))
    
    def select(self,text):
        def cond(key):
            data=key.split("_")[0]
            return data==text
        return [value_i
                   for key_i,value_i in self.items()
                       if(cond(key_i))]

    def by_data(self):
        data_dict={}
        for data_i in self.datasets():
            data_dict[data_i]=self.select(data_i)
        return data_dict

    @classmethod 
    def read(cls,in_path):
        paths=utils.bottom_files(in_path)
        shap_matrices={}
        for path_i in paths:
            id_i=path_i.replace(in_path,"")
            data,clf=id_i.split("/")
            shap_i=ShapleyMatrix( data=data,
                                  clf=clf,
                                  matrix=read(path_i))
            shap_matrices[shap_i.id]=shap_i
        return cls(shap_matrices)

class ShapleyMatrix(object):
    def __init__( self,
                  data,
                  clf,
                  matrix):
        self.data=data 
        self.clf=clf
        self.id=f"{data}_{clf}" 
        self.matrix=matrix
    
    def density(self):
        points=self.matrix.flatten()
        points-=np.mean(points)
        points/=np.std(points)
        return dist.EmpDist.make(points)
    
    def to_arr(self):
        return self.matrix.flatten()

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

def show_shapley( conf_path,
                  img_path,
                  plot_type):
    shap_matrices=get_shapley(conf_path)
    data_shape=shap_matrices.by_data()
    def helper(data_i,value_i):
        return dist.plot_multi( data_i,
                              value_i,
                              plot_type=plot_type)
    plot_decor( iter=data_shape.items(),
                fun=helper,
                out_path=img_path)

def plot_decor( iter,
                fun,
                out_path):
    if(out_path):
        utils.make_dir(out_path)
    for id_i,value_i in iter:
        plot=fun(id_i,value_i)
        if(out_path):
            out_i=f"{out_path}/{id_i}"
            plot.savefig( out_i,
                         dpi=300, 
                         bbox_inches="tight")
            plot.clf()
        else:
            plot.show()

def _shap_matrix(shap_i,out_i):
    plot.show_heatmap( shap_i.matrix,
                       shap_i.id,
                       out_i)

def _dist_matrix( shap_i,
                 out_i,
                 plot_type="fourrier"):
    dist_i=shap_i.density()
    plt=dist_i.plot( shap_i.id,
                     type=plot_type)
    plt.savefig( out_i,
                 dpi=300, 
                 bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="uci/conf.json")
    parser.add_argument("--img_path",type=str,default=None) 
    parser.add_argument("--img_type",type=str,default="dist") 
    parser.add_argument("--cmd", type=str,default="show")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        compute_shapley(args.conf_path)
    if(args.cmd=="show"):
        show_shapley( args.conf_path,
                      args.img_path,
                      args.img_type)