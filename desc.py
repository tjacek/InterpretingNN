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

def get_arg_dicts(data_path,matrix_path):
    data_dict= dataset.get_data_dict(data_path)
    matrix_dict=shape.get_matrix_dict(matrix_path)
    args_dicts = [  { "id":key_i,
                      "data":data_dict[key_i],
                      "shape":matrix_dict[key_i] } 
                   for key_i in data_dict]
    return args_dicts

def make_desc(data_path,matrix_path):
    args_dict=get_arg_dicts(data_path,matrix_path)
    features= GruopOfFeature([Basic(),PcaFeats()])
    df=dataset.make_df(helper=features,
                       iterable=args_dict,
                       cols=features.names(),
                       multi=False)
    print(df)

def _make_desc(in_path):
    lines=[]
    for id_i,path_i in utils.iter_files(in_path):
        values = np.loadtxt(path_i)
        values=np.abs(values)
        values/=np.sum(values,axis=0)
        values=np.amax(values,axis=0)
        values= np.round(values,4)
        gini_i=utils.gini(values)
        if(gini_i>0.0):
            print(id_i)
            print(values)
            print(gini_i)
 


def cls_gini(values):
    return np.array([ utils.gini(v_i) for v_i in values.T])
    
def max_mean(values):
    return np.array([ np.amax(v_i)/np.mean(v_i) 
                      for v_i in values.T])

make_desc(["AutoML/data","uci/data"], "shapley")