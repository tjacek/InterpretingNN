import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

class SplitProtocol(object):
    def __init__(self,n_splits,n_repeats):
        self.n_splits=n_splits
        self.n_repeats=n_repeats

    def get_split(self,data):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                     n_splits=self.n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(Split(train_index,test_index))
        return splits

class Split(object):
    def __init__(self,train_index,test_index):
        self.train_index=train_index
        self.test_index=test_index

        
    def eval(self,data,clf):
        if(type(clf)==str):
            clf=get_clf(clf)
        return data.eval(train_index=self.train_index,
                         test_index=self.test_index,
                         clf=clf)
       
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

def random_split(n_samples,p=0.9):
#    if(type(n_samples)==dataset.Dataset):
#        n_samples=len(n_samples)
    train_index,test_index=[],[]
    for i in range(n_samples):
        if(np.random.rand()<p):
            train_index.append(i)
        else:
            test_index.append(i)
    return Split(train_index=np.array(train_index),
                 test_index=np.array(test_index))