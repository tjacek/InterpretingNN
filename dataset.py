import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]
        
    def n_cats(self):
        return int(max(self.y))+1

    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def eval(self,train_index,test_index,clf,as_result=True):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf)
        return result,history
    
    def pred(self,test_index,clf):
        if(test_index is None):
            X_test,y_test=self.X,self.y
        else:
            X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        return Result(y_pred,y_test)

    def params_dict(self):
        return { "n_cats":self.n_cats(),
                 "dims":(self.dim(),),
                 "n_samples":len(self)}
#                "class_weight":self.weight_dict()}

    def range(self):
        return [ (np.amin(x_i),np.amax(x_i))
                    for x_i in self.X.T]

    def class_sizes(self):
        cats=  list(set(self.y))
        n_cats= len(cats) 
        params={cat_i:0 for cat_i in cats}
        for y_i in self.y:
            params[y_i]+=1
        return params

    def IR(self):
        sizes=self.class_sizes()
        values=list(sizes.values())
        return max(values)/min(values)
    
    def pca_feats(self):
        pca = PCA(n_components=None)
        pca.fit(self.X)
        exp_var=pca.explained_variance_ratio_
        greatest=exp_var[0]
        cum_var=np.cumsum(exp_var)
        int_var=(cum_var>0.95).astype(int)
        thres_var= np.argmax(int_var)/cum_var.shape[0]
        return greatest,thres_var

class Split(object):
    def __init__(self,train_index,test_index):
        self.train_index=train_index
        self.test_index=test_index
    
    def fit_clf(self,data,clf):
        return data.fit_clf(self.train_index,clf)

    def pred(self,data,clf):
        return data.pred(self.test_index,
                         clf=clf)

    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

    def __str__(self):
        train_size=self.train_index.shape[0]
        test_size=self.test_index.shape[0]
        return f"train:{train_size},test:{test_size}"     
    
    @classmethod
    def random(cls,n_samples,p=0.9):
        if(type(n_samples)==Dataset):
            n_samples=len(n_samples)
        train,test=[],[]
        for i in range(n_samples):
            if(np.random.rand()<p):
                train.append(i)
            else:
                test.append(i)
        train,test=np.array(train),np.array(test)
        return cls( train_index=train,
                    test_index=test)
class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)
  
    def save(self,out_path):
        y_pair=np.array([self.y_pred,self.y_true])
        np.savez(out_path,y_pair)
    
    @classmethod
    def read(cls,in_path:str):
        if(type(in_path)==Result):
            return in_path
        raw=list(np.load(in_path).values())[0]
        y_pred,y_true=raw[0],raw[1]
        return cls( y_pred=y_pred,
                    y_true=y_true)

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    if(type(in_path)!=str):
        return in_path
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

if __name__ == '__main__':
    data=read_csv("uci/satimage")
    print(data.pca_feats())