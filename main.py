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
#        print(in_i)

#print(clfs.TYPES)
make_splits("selected/data",
	        "selected/output")