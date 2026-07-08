import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import base,dataset, utils

def compute_shapley(in_path):
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        split_i=base.Split.random(data_i,0.9)
        clf_i=RandomForestClassifier()
        split_i.fit_clf(data_i,clf_i)
        explainer = shap.Explainer(clf_i.predict_proba, 
         	                       data_i.X[split_i.train_index])
        shap_values = explainer(data_i.X[split_i.test_index])
        values=np.sum(shap_values.values,axis=0)
        print(path_i)
        print(values.shape)
        print(values[0])
#        raise Exception(type(shap_values))

compute_shapley("uci/data")	