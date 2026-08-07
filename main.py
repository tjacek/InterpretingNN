import numpy as np
import argparse
import base,clfs,dataset,utils

SPLITS_DIRNAME = "splits"

class DirProxy(object):
	def __init__(self,dir_path,clf):
		self.path=f"{dir_path}/{clf}"
		utils.make_dir(self.path)
		self.files={}

	def _make(self,key):
		if(not key in self.files):
			path=f"{self.path}/{key}"
			utils.make_dir(path)
			self.files[key]=path
		return self.files[key]
    
	@property
	def ablat(self):
		return self._make("ablat")

	@property
	def results(self):
		return self._make("results")

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
#train("selected/data",
#	        "selected/output")
ablat_matrix("selected/output")