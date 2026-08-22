import numpy as np
import shap
from tqdm import tqdm
import argparse,os.path
import exp
import base,main,plot,utils

def compute_shapley(shap_exp):
    data,splits=shap_exp.get_data()
    clf_type=shap_exp.get_clf()
    def helper(split_i,clf_i):
        train,test=data.divide(split_i)
        if(shap_exp.k is None):
            background_data=train.X
        else:
            kmeans_summary = shap.kmeans( train.X, 
                                          shap_exp.k)
            background_data = kmeans_summary.data     
        explainer=shap.Explainer( clf_i.proba_fun(),
                                  train.X)
        shap_values = explainer(test.X)#,max_evals=620)
        return shap_values.values
    print(shap_exp.out_path)
    utils.make_dir(shap_exp.out_path)
    for i,split_i in enumerate(tqdm(splits)):
        out_i=f"{shap_exp.out_path}/{i}"
        if os.path.exists(out_i+".npz"):
            continue
        clf_i,_=split_i.fit_clf(data,clf_type())
        values_i=helper(split_i,clf_i)
        np.savez(out_i, values_i)

def show_shapley( in_path,
                  regex=r'^shapley(.)+'):
    conf_dict=utils.read_json(in_path)
    paths=utils.find_paths( conf_dict["out_path"],
                            regex=regex)
    for path_i in paths:
        if(os.path.exists(path_i)):
             matrix=get_matrix(path_i)
             plot.show_heatmap(matrix,path_i.split("/")[-1])

def get_matrix(shap_path):
    all_shape=[]
    for path_i in utils.top_files(shap_path):
        shape_i=np.load(path_i)["arr_0"]
        all_shape.append(shape_i)
    shap_arr=np.concatenate(all_shape,axis=0)
    shap_matrix=np.mean(shap_arr,axis=0)
    return shap_matrix

def k_exp(shap_exp):
    values=[50,100,200]
    k_iter=shap_exp.iter_exp("k",values)
    for exp_i in k_iter:
        exp_i.out_path+=f"{exp_i.k}"
        print(exp_i.out_path)
        compute_shapley(exp_i)

def var_matrix( in_path,
                regex=r'^shapley(.)+'):
    conf_dict=utils.read_json(in_path)
    paths=utils.find_paths( conf_dict["out_path"],
                            regex=regex)
    indiv_matrix=[ get_matrix(path_i) for path_i in paths]
    indiv_matrix=np.array(indiv_matrix)
    mean_matrix=np.mean(indiv_matrix,axis=0)
    plot.show_heatmap(mean_matrix,"mean")
    std_matrix=np.std(indiv_matrix,axis=0)
    plot.show_heatmap(std_matrix,"std")

def shapley_exp(in_path):
    conf=utils.read_json(in_path)
    prototype=exp.ExpParams( conf["data_path"],
                             conf["split_path"],
                             conf["out_path"])
    clf_iter=prototype.iter_exp("clf_type",conf["clf"])
    utils.make_dir(prototype.out_path)
    for exp_i in clf_iter:
        k_iter=exp_i.iter_exp( "k",conf["k"])
        for exp_j in k_iter:
            exp_j.out_path+=f"{exp_j.clf_type}_{exp_j.k}"
            compute_shapley(exp_j)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="selected/conf.json") 
    parser.add_argument("--regex", type=str,default=r"MLP_(.)+")
    parser.add_argument("--cmd", type=str,default="compute")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        shapley_exp(args.conf_path)
    if(args.cmd=="show"):
        show_shapley(  args.conf_path,
                       regex=args.regex)
    if(args.cmd=="var"):
        var_matrix( args.conf_path,
                    regex=args.regex)