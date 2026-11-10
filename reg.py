import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass,field
import dataset 

class DescDataset(object):
    def __init__( self,
		          df,
		          target="target"):
        self.df=df
        self.target=target
    
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

class RegAlg:
    def __init__(self):
        self.reg= None

    def exp(self,train,test):
        self.reg = LinearRegression()
        self.reg.fit(train.X,train.y)
        y_pred=self.reg.predict(test.X)
        return y_pred,test.y

    def show(self,feats):
        for feat_i,coef_i in zip(feats,self.reg.coef_):
            print(feat_i)
            print(coef_i)

def regression(desc_path):
    df=pd.read_csv(desc_path)
    desc=DescDataset(df)
    alg=RegAlg()
    names,y_true,y_pred=[],[],[]
    for pairs_i,data_i in desc.leve_one_out():
        pairs_i.norm()
        pred_i,true_i=alg.exp(pairs_i.train,
                              pairs_i.test)
        names.append(data_i)
        y_true.append(true_i)
        y_pred.append(pred_i[0])
    output=RegOutput( names,
                      np.array(y_true),
                      np.array(y_pred))
    print(output.as_df())
    print(output.mse())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_type", type=str,default="logistic")
    parser.add_argument("--desc", type=str,default="uci/desc.txt")
    args=parser.parse_args()
    regression( args.desc)