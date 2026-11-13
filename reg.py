import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from dataclasses import dataclass,field
import dataset 

class DescDataset(object):
    def __init__( self,
		          df,
		          target="target"):
        self.df=df
        self.target=target

    @classmethod
    def read(cls,desc_path):
        return cls(pd.read_csv(desc_path))
    
    def feats(self):
        taboo=set(["data",self.target])
        return [ col_i
		         for col_i in self.df.columns
		             if(not col_i in taboo)]
    
    def to_array(self):
        feat_df=self.df[self.feats()]
        X=feat_df.to_numpy()
        y=self.df[self.target].to_numpy()
        return X,y


    def leve_one_out(self):
        make_pairs=dataset.DatasetPair.make
        self.df = self.df.reset_index(drop=True)
        for i in range(len(self.df)):
            train_df = self.df.drop(i)
            test_df = self.df.iloc[[i]]
            cols=[self.target, "data"]
            X_train = train_df.drop(columns=cols).to_numpy()
            y_train = train_df[self.target].to_numpy()
        
            X_test = test_df.drop(columns=cols).to_numpy()
            y_test = test_df[self.target].iloc[0]
            
            pairs=make_pairs(X_train, y_train,
                       X_test, y_test)
            data_i = test_df["data"].iloc[0]
            yield pairs,data_i 

@dataclass
class RegOutput(object):
    names: list
    y_true: list 
    y_pred: list 
    
    def mse(self):
        error=self.y_true - self.y_pred
        print(error**2)
        return np.sqrt(np.mean(error**2))
    
    def as_df(self):
        return pd.DataFrame.from_dict(self.__dict__)

    def show(self):
        for i,name_i in enumerate(self.names):
            pred_i=self.y_pred[i]
            true_i=self.y_true[i]
            print(f"{name_i},pred:{pred_i:.4f},true_i{true_i:.4f}")

class LinearAlg:
    def __init__(self):
        self.reg= LinearRegression()
    
    def fit(self,X,y):
        self.reg.fit(X,y)
    
    @classmethod
    def exp(cls,train,test):
        reg=cls()
        reg.fit(train.X,train.y)
        y_pred=reg.reg.predict(test.X)
        return reg,(y_pred[0],test.y)

    def show(self,feats):
        for feat_i,coef_i in zip(feats,self.reg.coef_):
            print(f"{feat_i},{coef_i:.4f}")

class GaussAlg:
    def __init__(self):
        kernel =  RBF(length_scale=1.0, 
                      length_scale_bounds=(1e-2, 1e2))
        self.reg = GaussianProcessRegressor(kernel=kernel, 
                                            n_restarts_optimizer=9)
    def fit(self,X,y):
        self.reg.fit(X,y)
    
    @classmethod
    def exp(cls,train,test):
        reg=cls()
        reg.fit(train.X,train.y)
        y_pred=reg.reg.predict(test.X)
        return reg,(y_pred[0],test.y)

def regression(desc_path):
    desc=DescDataset.read(desc_path)
    alg=GaussAlg
    names,models,results=[],[],[]
    for pairs_i,data_i in desc.leve_one_out():
        pairs_i.norm()
        model_i,result_i=alg.exp(pairs_i.train,
                              pairs_i.test)
        names.append(data_i)
        models.append(model_i)
        results.append(result_i)
    y_pred,y_true=list(zip(*results))
    output=RegOutput( names,
                      np.array(y_true),
                      np.array(y_pred))
    print(output.as_df())
    print(output.mse())

def linear_reg(desc_path):
    desc=DescDataset.read(desc_path)
    alg=RegAlg()
    X,y=desc.to_array()
    alg.fit(X,y)
    alg.show(desc.feats())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_type", type=str,default="logistic")
    parser.add_argument("--desc", type=str,default="uci/desc.txt")
    args=parser.parse_args()
    regression( args.desc)