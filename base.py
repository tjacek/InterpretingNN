import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import utils

class SplitGroup(object):
    def __init__(self,splits,n_repeats=1):
        self.splits=splits
        self.n_repeats=n_repeats
    
    def __len__(self):
        return len(self.splits)

    def __call__(self,data,clf_type):
        all_results=[]
        for split_i in tqdm(self.splits):
            clf_i=clf_type()
            split_i.fit_clf(data,clf_i)
            result_i=split_i.pred(data,clf_i)
            all_results.append(result_i)
        return ResultGroup(all_results)

    @classmethod
    def make( cls,
              data,
              n_repeats=1,
              n_splits=10):
        rskf=RepeatedStratifiedKFold(n_repeats=n_repeats, 
                                     n_splits=n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(Split(train_index,test_index))
        return cls(splits,n_repeats)
    
    def save(self,out_path):
        utils.make_dir(out_path)
        for i,split_i in enumerate(self.splits):
            split_i.save(f"{out_path}/{i}")

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
    
    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

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
    
    def save(self,out_path):
        y_pair=np.array([self.y_pred,self.y_true])
        np.savez(out_path,y_pair)

class ResultGroup(object):
    def __init__(self,indiv_result):
        self.indiv_result=indiv_result

    def get_acc(self):
        acc=[ result_i.get_acc() 
                for result_i in self.indiv_result]
        return np.mean(acc)

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,result_i in enumerate(self.indiv_result):
            result_i.save(f"{out_path}/{i}")
