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
        explainer = shap.Explainer(clf_i.predict_proba, 
         	                       data_i.X[split_i.train_index])
        shap_values = explainer(data_i.X[split_i.test_index])
        values=np.mean(np.abs(shap_values.values),axis=0)

        sn.heatmap(values,cmap="YlGnBu",#linewidths=0.5,
                   annot=True,annot_kws={"size": 5}, fmt='g')
        id_i=path_i.split("/")[-1]
        plt.title(id_i)
        if(out_path):
            plt.savefig(f'{out_path}/{id_i}.png')
        else:
            plt.show()


compute_shapley("uci/data","shapley")	