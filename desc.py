import numpy as np
from abc import ABC, abstractmethod
import dataset,utils

class GruopOfFeature(object):
    def __init__(self,features):
        self.features=features

    def names(self):
        names=[]
        for feature_i in self.features:
            names_i= feature_i.names()   
            if(not isinstance(names_i, list)):
                names_i=[names_i]
            names.extend(names_i)
        return names

    def __call__(self,data):
        values=[]
        for feature_i in self.features:
            values_i= feature_i(data)   
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


def make_desc(in_path):
    lines=[]
    for id_i,path_i in utils.iter_files(in_path):
        values = np.loadtxt(path_i)
        gini_i=max_mean(values)
        print(id_i)
        print(gini_i)

def cls_gini(values):
    return np.array([ utils.gini(v_i) for v_i in values.T])
    
def max_mean(values):
    return np.array([ np.amax(v_i)/np.mean(v_i) 
                      for v_i in values])

make_desc("shapley")