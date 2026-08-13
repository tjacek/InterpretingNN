import numpy as np
import pandas as pd
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
            path=self.subpath(key)
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

class PredDict(dict):
    def __getitem__(self,clf):
        clf_dict={}
        for name_i,df_i in self.items():
            acc = df_i.set_index("clf")["norm_acc"]
            clf_dict[name_i]=acc[clf]
        return clf_dict

    def issubset(self, names):
        return set(names).issubset(self.df_dict.keys())

    @classmethod
    def from_dir(cls,in_path):
        reader=base.ResultGroup.read
        lines=[]
        for id_i,clf_j,path_j in clf_dir_iter(in_path):
            result_j=reader(f"{path_j}/results")
            lines.append((id_i,clf_j,result_j.acc))
        df=pd.DataFrame.from_records(lines,
                                     columns=["data","clf","acc"])
        return cls.from_df(df)

    @classmethod
    def from_df(cls,df):
        clf_dfs={}
        for data_i in df["data"].unique():
            df_i = df[df["data"] == data_i].copy()             
            acc_i=df_i["acc"].to_list()
            min_i=min(acc_i)
            delta_i= max(acc_i)-min_i
            df_i["norm_acc"]=df_i["acc"].apply(lambda acc: (acc-min_i)/delta_i)
            df_i.sort_values(by="norm_acc",inplace=True)
            clf_dfs[data_i]=df_i
        return cls(clf_dfs)

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
    for id_i,clf_j,path_j in clf_dir_iter(out_path):
        result_j=base.ResultGroup.read(f"{path_j}/results")
        score=result_j.get_score(score_type)
        print(f"{id_i},{clf_j},{score:.4f}")

def clf_dir_iter(out_path):
    taboo=set(["ablat","splits"])
    for id_i,path_i in utils.iter_files(out_path):
        for clf_j,path_j in utils.iter_files(path_i):
            if(not clf_j in taboo):
                yield id_i,clf_j,path_j

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
    PredDict.from_dir("uci/output")
#ablat_matrix("selected/output",
#               clf_type="MLP")
