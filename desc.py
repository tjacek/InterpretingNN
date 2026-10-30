import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from dataclasses import dataclass
import argparse
import dataset
import shape
import train
import utils

@dataclass
class Sources:
    datasets:dict
    shapley:dict
    pred:dict

    @classmethod
    def make(cls,conf):
        paths=utils.iter_files(conf["data_path"])
        datasets={ id_i:dataset.read_csv(path_i) 
                   for id_i,path_i in paths}
        shapley=shape.ShapleyGroup.read(conf["out_path"])
        pred=train.get_pred(conf["result_path"],verbose=True)
        return cls( datasets,
                    shapley,
                    pred)

class GruopOfFeature(object):
    def __init__( self,
                  features,
                  sources):
        self.features=features
        self.sources=sources

    def names(self):
        names=["data"]
        for feature_i in self.features:
            names_i= feature_i.names()   
            if( isinstance(names_i, list)):
                names.extend(names_i)
            else:
                names.append(names_i)
        return names

    def data(self):
        return self.sources.datasets.keys()

    def __call__(self,id):
        values=[id]
        for feature_i in self.features:
            value_i=feature_i(id,self.sources)
            if(isinstance(value_i, list)):
                values.extend(value_i)
            else:
                values.append(value_i)
        return values    

class Feature(ABC):
    @abstractmethod
    def names(self):
        pass
   
    @abstractmethod
    def __call__(self,data_id,sources):
        pass

    def __repr__(self):
        names=self.names()
        if(type(names)==list):
            names="-".join(names)
        return names

class Basic(Feature):
    def names(self):
        return ["feats","samples", "classes"]

    def __call__(self,data_id,sources):
        data=sources.datasets[data_id]
        return data.params.to_list()

class Target(Feature):
    def __init__( self,
                  clf_x="RF",
                  clf_y="TabPFN"):
       self.clf_x=clf_x
       self.clf_y=clf_y

    def names(self):
        return "target"
    
    def __call__(self,data_id,sources):
        diff=sources.pred.diff( self.clf_x, 
                                self.clf_y)
        return diff[data_id]

class IR(Feature):
    def names(self):
        return "IR"

    def __call__(self,data_id,sources):
        data=sources.datasets[data_id]
        return data.params.IR()

class Shapley(Feature):
    def __init__(self,clf="RF"):
        self.clf=clf

    def names(self):
        return f"shapley({self.clf})"

    def __call__( self,
                  data_id,
                  sources):
        shap=sources.shapley
        shap_matrix=shap.select(self.clf).by_id(data_id)
        dist= shap_matrix.density()
        return dist.kurt()

class GINI(Feature):
    def names(self):
        return "gini"

    def __call__( self,
                  data_id,
                  sources):
        data=sources.datasets[data_id]
        return gini(data.params.sizes())

def gini(arr):
    arr = np.sort(np.asarray(arr))
    n = len(arr)
    total = np.sum(arr)
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return np.sum((2*index - n - 1)*arr) / (n*total)


def make_desc( conf_path,
               features_list=None):
    conf_dict=utils.read_json(conf_path)
    sources= Sources.make(conf_dict)
    if(features_list is None):
        features_list=[ Basic(),IR(),GINI(),Shapley(),
                        Target()]
    features= GruopOfFeature(features_list,sources)
    df=dataset.make_df( helper=features,
                        iterable=features.data(),
                        cols=features.names(),
                        multi=False)
    print(df.round(4))
#def make_desc( conf_path,
#               features_list=None,
#               order_by="feats",
#               out_path=None):
#    conf_dict=utils.read_json(conf_path)
#    args_dict=get_arg_dicts(conf_dict["data"],
#                             conf_dict["sources"])
#    if(features_list is None):
#        features_list=[Basic(),PcaFeats(),IR(),GINI()]
#    features= GruopOfFeature(features_list)
#    df=dataset.make_df( helper=features,
#                        iterable=args_dict,
#                        cols=features.names(),
#                        multi=False)
#    df.sort_values(by=order_by,inplace=True)
#    print(df.round(4))
#    if(out_path):
#        df.to_csv(out_path, sep=",", index=False)

def plot_xy(x_path,y_path):
    x_dict=plot.get_matrix_dict(x_path)
    y_dict=plot.get_matrix_dict(y_path)
    for key_i in x_dict:
        x_i=x_dict[key_i].flatten()
        y_i=y_dict[key_i].flatten()
        plot.plot_xy( x_i,y_i,
                      title=key_i, 
                      x_label=x_path,
                      y_label=y_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str,default="uci/conf.json")
    parser.add_argument("--order", type=str,default="Infl(shapley)")
    parser.add_argument("--out", type=str,default=None)#"desc/infl3")
    args=parser.parse_args()
    make_desc( args.conf)#,
#               features_list=features_list,
#               order_by=args.order,
#               out_path=args.out)
