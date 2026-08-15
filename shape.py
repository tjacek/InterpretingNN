import numpy as np
import shap
from dataclasses import dataclass
import argparse,os.path
import clfs
import dataset
import main,plot,utils

@dataclass
class ShapleyExp:
    data_path:str      
    split_path:str
    clf_type:str
    out_path:str 
    k:int
    
    def get_data(self):
        data=dataset.read_csv(self.data_path)
        splits=base.SplitGroup.read(self.split_path)
        return data,splits

    def get_clf(self):
        return clfs.TYPES[self.clf_type]

def compute_shapley(shap_exp):
    data,splits=shap_exp.get_data()
    clf_type=shap_exp.get_clf()
    for i,split_i in enumerate(splits):
        out_i=f"{shapley_path}/{i}"
        if os.path.exists(out_i+".npz"):
            continue
        clf_i,_=split_i.fit_clf(data,clf_type())
        values_i=shape_split( data,
                              split_i,
                              clf_i
                              k=shap_exp.k)
        np.savez(out_i, values_i)

def _compute_shapley(  data_path,
	                  dir_path,
	                  clf="TabPNF"):
    dir_proxy=main.DirProxy(dir_path,clf)
    clf_type=clfs.TYPES[clf]
    data=dataset.read_csv(data_path)
    shapley_path=dir_proxy.dispatch("shapley")
    splits=dir_proxy.get_splits()
    for i,split_i in enumerate(splits):
        out_i=f"{shapley_path}/{i}"
        print(out_i)
        if os.path.exists(out_i+".npz"):
            continue
        clf_i,_=split_i.fit_clf(data,clf_type())
        values_i=shape_split(data,split_i,clf_i)
        np.savez(out_i, values_i)

def shape_split( data_i,
                 split_i,
                 clf_i,
                 k=100):
    train,test=data_i.divide(split_i)
    print(model_i.NAME)
#    explainer=model_i.get_explainer(train.X)
    if(k is None):
        background_data=train.X
    else:
        kmeans_summary = shap.kmeans(train.X, k)
        background_data = kmeans_summary.data     
    shap.Explainer( clf_i.model.predict_proba, 
                    train.X)
    shap_values = explainer(test.X)#,max_evals=620)
    return shap_values.values

def show_shapley(dir_path):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,default="selected/data/cleveland")
    parser.add_argument("--dir_path", type=str,default="selected/output/cleveland")
    parser.add_argument("--clf", type=str,default="TabPNF")
    parser.add_argument("--cmd", type=str,default="compute")
    args=parser.parse_args()
    if(args.cmd=="compute"):
         compute_shapley( args.data_path,
	                      args.dir_path,
	                      args.clf)
    if(args.cmd=="show"):
        show_shapley(  args.dir_path)