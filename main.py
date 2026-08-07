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

#print(clfs.TYPES)
train("selected/data",
	        "selected/output")