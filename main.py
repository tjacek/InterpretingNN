import argparse
import base,clfs,dataset,utils


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
        splits_i.save(f"{out_i}/splits")

def ablation( in_path,
	          out_path,
	          clf_type="RF"):
    clf_type=clfs.TYPES[clf_type]
    for in_i,out_i in utils.out_iter(in_path,
    	                             out_path):
        data_i=dataset.read_csv(in_i)
        splits_i=base.SplitGroup.read(f"{out_i}/splits")
        ablat_i=f"{out_i}/ablat"
        utils.make_dir(ablat_i)
        print(in_i)
        for j in range(data_i.dim()):
            data_j=data_i.remove_col(j)
            result_j,_=splits_i(data_j,clf_type)
            result_j.save(f"{ablat_i}/{j}")

#print(clfs.TYPES)
ablation("selected/data",
	        "selected/output")