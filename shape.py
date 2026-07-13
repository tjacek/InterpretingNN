import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score
import base,dataset, utils

def compute_shapley(in_path,out_path=None):
    if(out_path):
        utils.make_dir(out_path)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        split_i=base.Split.random(data_i,0.9)
        clf_i=RandomForestClassifier()
        split_i.fit_clf(data_i,clf_i)
        train,test=data_i.divide(split_i)

        explainer = shap.Explainer(clf_i.predict_proba, 
         	                      train.X)
        shap_values = explainer(test.X,max_evals=620)
        values=np.mean(np.abs(shap_values.values),axis=0)
        if(out_path):
            id_i=path_i.split("/")[-1]
            np.savetxt(f'{out_path}/{id_i}.txt', values, fmt='%f')

def show_heatmap(in_path):
    for path_i in utils.top_files(in_path):
        values = np.loadtxt(path_i)
        sn.heatmap(values,cmap="YlGnBu",linewidths=0.5,
                   annot=True,annot_kws={"size": 5}, fmt='g')
        id_i=path_i.split("/")[-1]
        plt.title(id_i)
        plt.show()


compute_shapley("AutoML/_data","shapley")	
#show_heatmap("shapley")