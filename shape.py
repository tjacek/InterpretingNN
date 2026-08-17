import numpy as np
import shap
from tqdm import tqdm
from dataclasses import dataclass,field
import argparse,os.path
import clfs
import dataset
import base,main,plot,utils

@dataclass
class ShapleyExp:
    data_path:str      
    split_path:str
    out_path:str = field(default=None)
    clf_type:str = field(default='RF')
    k:int = field(default=100)
    
    def get_data(self):
        data=dataset.read_csv(self.data_path)
        splits=base.SplitGroup.read(self.split_path)
        return data,splits

    def get_clf(self):
        return clfs.TYPES[self.clf_type]

    def iter_exp(self,attr,values):
        for value_i in values:
            exp_i = ShapleyExp(**self.__dict__)
            setattr(exp_i, attr, value_i)
            yield exp_i

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
        explainer=shap.Explainer( clf_i.proba_fun(),#model.predict_proba, 
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

def show_shapley(dir_path):
#    utils,
    for dir_i in main.DirProxy.all_clfs(dir_path):
        shap_path=dir_i.subpath(dir_i.SHAP)
        if(os.path.exists(shap_path)):
             matrix=get_matrix(shap_path)
             plot.show_heatmap(matrix,dir_i.clf)

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
#    raise Exception(paths)
    indiv_matrix=[ get_matrix(path_i) for path_i in paths]
    indiv_matrix=np.array(indiv_matrix)
    mean_matrix=np.mean(indiv_matrix,axis=0)
    plot.show_heatmap(mean_matrix,"mean")
    std_matrix=np.std(indiv_matrix,axis=0)
    plot.show_heatmap(std_matrix,"std")

def inf_exp(in_path):
    conf_dict=utils.read_json(in_path)
    prototype=ShapleyExp(conf_dict["data_path"],
                         conf_dict["split_path"],
                         conf_dict["out_path"])
    clf_iter=prototype.iter_exp("clf_type",conf_dict["clf"])
    for exp_i in clf_iter:
        k_iter=exp_i.iter_exp( "k",
                               conf_dict["k"])
        for exp_j in k_iter:
            exp_j.out_path+=f"{exp_j.clf_type}_{exp_j.k}"
            compute_shapley(exp_j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="selected/conf.json") 
    parser.add_argument("--regex", type=str,default=r"MLP_(.)+")
    parser.add_argument("--cmd", type=str,default="var")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        inf_exp(args.conf_path)
#    if(args.cmd=="show"):
#        show_shapley(  args.dir_path)
    if(args.cmd=="var"):
        var_matrix( args.conf_path,
                    regex=args.regex)
#    if(args.cmd=="compute"):
#        utils.make_dir(f"{args.dir_path}/{args.clf}")
#        exp=ShapleyExp( args.data_path,      
#                        f"{args.dir_path}/splits",
#                        args.clf,
#                        f"{args.dir_path}/{args.clf}/shapley",
#                        k=None)
#        k_exp(exp)
    