import numpy as np
import pandas as pd
from sklearn import preprocessing
from dataclasses import dataclass
import base,utils

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y
        self._params=None

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
        return base.Result(y_pred,y_test)

    def params(self):
        if(self._params is None):
            self._params= DatasetParams( self.dim(),
                                         len(self),
                                         self.n_cats(),
                                         self.class_sizes())
        return self._params

    def range(self):
        return [ (np.amin(x_i),np.amax(x_i))
                    for x_i in self.X.T]

    def norm(self):
        scaler=preprocessing.RobustScaler()
        self.X= scaler.fit_transform(self.X)
        return self

    def class_sizes(self):
        cats=  list(set(self.y))
        n_cats= len(cats) 
        params={cat_i:0 for cat_i in cats}
        for y_i in self.y:
            params[y_i]+=1
        return params
    
    def divide(self,split):
        train=self.select(split.train_index)
        test=self.select(split.test_index)
        return train,test
    
    def select(self,index):
        return Dataset(self.X[index],
                       self.y[index])

    def remove_col(self,i):
        return Dataset(X=np.delete(self.X,[i],1),
                       y=self.y)

    def binarize(self,i):
        binary_y=[ int(y_i==i) 
                    for y_i in self.y]
        return Dataset( X=self.X,
                        y=np.array(binary_y))
@dataclass
class DatasetParams:
    feats:int
    samples:int
    cats:int
    sizes_dict:dict
    
    @classmethod
    def from_arr(self,X,y):
        return Dataset(X,y).params()

    def to_list(self):
        return [self.feats,self.samples,self.cats]

    def sizes(self):
        return list(self.sizes_dict.values())

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

def make_df(helper,
            iterable,
            cols,
            multi=False):
    lines=[]
    if(multi):
        for arg_i in iterable:
            lines+=helper(arg_i)
    else:
        for arg_i in iterable:
            lines.append(helper(arg_i))
    df=pd.DataFrame.from_records(lines,
                                columns=cols)
    return df 

def get_data_dict(in_path):
    return { id_i:read_csv(path_i) 
                for id_i,path_i in utils.iter_files(in_path)}

def make_desc(in_path,verbose=True):
    paths=list(utils.iter_files(in_path))
    def helper(args):
        id_i,path_i=args
        data_i=read_csv(path_i)
        line_i=[ id_i,
                 data_i.n_cats(),
                 data_i.dim(),
                 len(data_i),
                 round(data_i.IR(),2)]
        line_i+=data_i.pca_feats()
        return line_i
    cols=["data","classes","feats","samples",
          "IR","pca_max","pca_95"]
    df=make_df(helper,
               iterable=paths,
               cols=cols)
    if(verbose):
        print(df)
    return df

if __name__ == '__main__':
    make_desc("spatial")