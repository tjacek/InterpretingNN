import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import dataset,shape,utils

class GruopOfFeature(object):
    def __init__(self,features):
        self.features=features

    def names(self):
        names=["data"]
        for feature_i in self.features:
            names_i= feature_i.names()   
            if(not isinstance(names_i, list)):
                names_i=[names_i]
            names.extend(names_i)
        return names

    def __call__(self,arg_dict):
        values=[arg_dict["id"]]
        for feature_i in self.features:
            values_i= feature_i(arg_dict)   
            if(not isinstance(values_i, list)):
                values_i=[values_i]
            values.extend(values_i)
        return values

class Feature(ABC):
    def names(self):
        return str(self)
   
    @abstractmethod
    def __call__(self,data):
        pass

class Basic(Feature):
    def names(self):
        return ["classes", "feats", "samples"]

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        data_params=data.params()
        return data_params.to_list()

class PcaFeats(Feature):
    def names(self):
        return ["pca_max","pca_95"]

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        pca = PCA(n_components=None)
        pca.fit(data.X)
        exp_var=pca.explained_variance_ratio_
        greatest=exp_var[0]
        cum_var=np.cumsum(exp_var)
        int_var=(cum_var>0.95).astype(int)
        thres_var= np.argmax(int_var)/cum_var.shape[0]
        return [greatest,thres_var]

class IR(Feature):
    def __str__(self):
        return "IR"

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        data_params=data.params()
        sizes=data_params.sizes()
        return max(sizes)/min(sizes)

class GINI(Feature):
    def __str__(self):
        return "Gini"

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        data_params=data.params()
        return utils.gini(data_params.sizes())

class Shapley(Feature):
    def __str__(self):
        return "Shapley"

    def __call__(self,arg_dict):
        values=arg_dict["shape"]
        values=np.abs(values)
        values/=np.sum(values,axis=0)
        values=np.amax(values,axis=0)
#        values= np.round(values,4)
        return utils.gini(values)

def get_arg_dicts(data_path,matrix_path):
    data_dict= dataset.get_data_dict(data_path)
    matrix_dict=shape.get_matrix_dict(matrix_path)
    args_dicts = [  { "id":key_i,
                      "data":data_dict[key_i],
                      "shape":matrix_dict[key_i] } 
                   for key_i in data_dict]
    return args_dicts

def make_desc( data_path,
               matrix_path,
               features_list=None,
               out_path=None):
    if(features_list is None):
        features_list=[Basic(),PcaFeats(),IR(),GINI(),Shapley()]
    args_dict=get_arg_dicts(data_path,matrix_path)
    features= GruopOfFeature(features_list)
    df=dataset.make_df(helper=features,
                       iterable=args_dict,
                       cols=features.names(),
                       multi=False)
    df.sort_values(by="feats",inplace=True)
    print(df)
    if(out_path):
        df.to_csv(out_path, sep=",", index=False)

import plot
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_xy(x_path,y_path):
    x_dict=plot.get_matrix_dict(x_path)
    y_dict=plot.get_matrix_dict(y_path)
    for key_i in x_dict:
        x_i=x_dict[key_i].flatten()
        y_i=y_dict[key_i].flatten()

        r, p = pearsonr(x_i, y_i)
        plt.figure(figsize=(12, 6))
        plt.scatter(x_i, y_i)
        plt.title(key_i)
        plt.xlabel(x_path)
        plt.ylabel(y_path)
        plt.figtext(
            0.5,
            0.01,
            f"Pearson correlation: r = {r:.4f}, p = {p:.3e}",
            ha="center",
            fontsize=11
        )
        plt.show()
#def cls_gini(values):
#    return np.array([ utils.gini(v_i) for v_i in values.T])
    
#def max_mean(values):
#    return np.array([ np.amax(v_i)/np.mean(v_i) 
#                      for v_i in values.T])

if __name__ == '__main__':
     plot_xy("matrix/infl","matrix/shapley")
#    features_list=[Basic(),IR(),GINI(),PcaFeats()]#,Shapley()]
#    make_desc( ["AutoML/data","uci/data"], 
#               "shapley",
#               features_list,
#               out_path="desc/pca")