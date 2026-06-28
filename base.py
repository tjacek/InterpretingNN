import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

class SplitGroup(object):
    def __init__(self,splits):#n_splits,n_repeats):
        self.splits=splits
 #       self.n_splits=n_splits
 #       self.n_repeats=n_repeats
    
    def __len__(self):
        return len(self.splits)

    @classmethod
    def get_split( cls,
                   data,
                   n_repeats=1,
                   n_splits=10):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                     n_splits=self.n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(Split(train_index,test_index))
        return cls(splits)

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