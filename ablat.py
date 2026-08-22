import numpy as np
import argparse
import base
import exp
import plot
import utils

def ablat_exp(in_path):
    conf=utils.read_json(in_path)
    proto=exp.ExpParams( conf["data_path"],      
                         conf["split_path"],
                         conf["out_path"])
    clf_type=proto.get_clf()
    utils.make_dir(proto.out_path)
    data,splits=proto.get_data()
    for i in range(data.dim()):
        data_i=data.remove_col(i)
        result_i,_=splits( data,
                           clf_type,
                           verbose=True)
        out_i=f"{proto.out_path}/{i}"
        result_i.save(out_i)

def ablat_matrix(in_path,score="acc"):
    matrix=[]
    for path_i in utils.top_files(in_path):
        result_i=base.ResultGroup.read(path_i)
        n_cats=result_i.n_cats()
        row_i=[ result_i.cat_score(j,score) 
                for j in range(n_cats)] 
        matrix.append(row_i)
    matrix=np.array(matrix)
    plot.show_heatmap(matrix,in_path.split("/")[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="selected/conf.json")
    parser.add_argument("--cmd", type=str,default="show")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        ablat_exp(args.conf_path)
    if(args.cmd=="show"):
        conf=utils.read_json(args.conf_path)
        ablat_matrix(conf["out_path"])