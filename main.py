import numpy as np
import argparse
import base,clfs,dataset,utils

SPLITS_DIRNAME = "splits"

class DirProxy(object):
    ABLAT="ablat"
    RESULT="results"
    SHAP="shapley"
    def __init__(self,dir_path,clf):
        self.path=f"{dir_path}/{clf}"
        self.clf=clf
        self.split_path=f"{dir_path}/{SPLITS_DIRNAME}"
        utils.make_dir(self.path)
        self.files={}
   
    def subpath(self,key):
        return f"{self.path}/{key}"

    def dispatch(self,key):
        if(not key in self.files):
            path=self.subpath(key) #f"{self.path}/{key}"
            utils.make_dir(path)
            self.files[key]=path
        return self.files[key]
    
    @property
    def ablat(self):
        return self.dispatch(self.ABLAT)

    @property
    def results(self):
        return self.dispatch(self.RESULT)

    @property
    def shapley(self):
        return self.dispatch(self.SHAP)
    
    def get_splits(self):
        return base.SplitGroup.read(self.split_path)
    
    @classmethod
    def all_clfs(cls,dir_path):
        return [  cls(dir_path,id_i)
                  for id_i,path_i in utils.iter_files(dir_path)
                      if(id_i!=SPLITS_DIRNAME)]

def make_splits( in_path,
	             out_path,
	             n_repeats=1,
                 n_splits=10):
    for in_i,out_i in utils.out_iter(in_path,
    	                             out_path):
        data_i=dataset.read_csv(in_i)
        splits_i=base.SplitGroup.make( data_i,
                                       n_repeats=n_repeats,
                                       n_splits=n_splits)
        utils.make_dir(out_i)
        splits_i.save(f"{out_i}/{SPLITS_DIRNAME}")

def split_iter(in_path,out_path):
    for in_i,out_i in utils.out_iter(in_path,
    	                             out_path):
        data_i=dataset.read_csv(in_i)
        splits_i=base.SplitGroup.read(f"{out_i}/{SPLITS_DIRNAME}")
        yield out_i,data_i,splits_i

def train( in_path,
	          out_path,
	          clf_type="RF"):
    clf=clfs.TYPES[clf_type]
    for out_i,data_i,splits_i in split_iter(in_path,out_path):
        dir_i=DirProxy(out_i,clf_type)
        result_i,_=splits_i(data_i,clf)
        result_i.save(f"{dir_i.results}")

def show_pred(out_path,score_type="f1"):
    taboo=set(["ablat","splits"])
    for id_i,path_i in utils.iter_files(out_path):
        for clf_j,path_j in utils.iter_files(path_i):
            if(not clf_j in taboo):
                result_j=base.ResultGroup.read(f"{path_j}/results")
                score=result_j.get_score(score_type)
                print(f"{id_i},{clf_j},{score:.4f}")

def ablation( in_path,
	          out_path,
	          clf_type="RF"):
    clf=clfs.TYPES[clf_type]
    for out_i,data_i,splits_i in split_iter(in_path,out_path):
    	dir_i=DirProxy(out_i,clf_type)
    	for j in range(data_i.dim()):
            data_j=data_i.remove_col(j)
            result_j,_=splits_i(data_j,clf)
            result_j.save(f"{dir_i.ablat}/{j}")

def ablat_matrix( result_dir,
	              clf_type="RF"):
    for path_i in utils.top_files(result_dir):
        dir_i=DirProxy(path_i,clf_type)
        ablat_f1=[]
        for path_i in utils.top_files(dir_i.ablat):
            result_i=base.ResultGroup.read(path_i)
            ablat_f1.append(result_i.f1)
        ablat_f1=np.array(ablat_f1)
        full_result=base.ResultGroup.read(dir_i.results)
        full_f1=full_result.f1
        print(ablat_f1-full_f1)

#print(clfs.TYPES)
#train( "selected/data",
#	   "selected/output",
#       clf_type="TabPNF")
if __name__ == '__main__':
    show_pred("selected/output")
#ablat_matrix("selected/output",
#               clf_type="MLP")
