from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import shap

class Clf(object):

    def fit(self,X,y):
        return self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)
    
    def __str__(self):
    	return self.NAME
    
    def __repr__(self):
    	return self.NAME

    def proba_fun(self):
        return self.model.predict_proba
    
    def get_explainer(self,train_x):
        return shap.Explainer( self.model.predict_proba, 
                               train_x)
class RF(Clf):
    NAME="RF"
    def __init__(self):
        self.model=RandomForestClassifier(class_weight="balanced") 
    
class GRAD(Clf):
    NAME="GRAD"
    def __init__(self):
        self.model=GradientBoostingClassifier()
    
class LR(Clf):
    NAME="LR"
    def __init__(self):
        self.model=LogisticRegression(solver='liblinear')

class SVM(Clf):
    NAME="SVM"
    def __init__(self):
        self.model=svm.SVC(kernel='rbf')