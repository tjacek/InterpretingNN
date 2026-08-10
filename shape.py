import shap
import clfs
import dataset
import main

def compute(  data_path,
	          dir_path,clf="MLP"):
    dir_proxy=main.DirProxy(dir_path,clf)
    splits=dir_proxy.get_splits()
    clf_type=clfs.TYPES["TabPNF"]
    data=dataset.read_csv(data_path)
    results,models=splits(data,clf_type)
    values=shape_split(data,splits.splits[0],models[0])
    print(values.shape)


def shape_split(data_i,split_i,model_i):
    train,test=data_i.divide(split_i)

    explainer = shap.Explainer(model_i.model.predict_proba, 
         	                   train.X)#,
#                                  algorithm="tree")
    shap_values = explainer(test.X,max_evals=620)
    return shap_values.values

compute("selected/data/cmc",
	    "selected/output/cmc",
	    "MLP")