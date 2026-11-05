import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass,field

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
        self.df = self.df.reset_index(drop=True)
        for i in range(len(self.df)):
            train_df = df.drop(i)
            test_df = df.iloc[[i]]

            X_train = train_df.drop(columns=[self.target, "data"]).to_numpy()
            y_train = train_df[self.target].to_numpy()
        
            X_test = test_df.drop(columns=[self.target, "data"]).to_numpy()
            y_test = test_df[self.target].iloc[0]
        
            train = dataset.Dataset(X_train, y_train)
            test = dataset.Dataset(X_test, y_test)

            data_i = test_df["data"].iloc[0]
#            if(norm):
#                scaler = StandardScaler()
#                train.X = scaler.fit_transform(train.X)
#                test.X = scaler.transform(test.X)
            yield train,test,data_i 

@dataclass
class RegOutput(object):
    names: list
    y_true: list 
    y_pred: list 
    
    def mse(self):
        error=self.y_true - self.y_pred
        print(error**2)
        return np.sqrt(np.mean(error**2))


def regression(desc_path):
    df=pd.read_csv(desc_path)
    desc=DescDataset(df)
    X,y=desc.to_array()
    reg = LinearRegression()
    reg.fit(X,y)
    for feat_i,coef_i in zip(desc.feats(),reg.coef_):
        print(feat_i)
        print(coef_i)
    y_pred=reg.predict(X)
    output=RegOutput( df["data"].to_list(),
                      y,
                      y_pred)
    print("****")
    print(output.mse())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_type", type=str,default="logistic")
    parser.add_argument("--desc", type=str,default="uci/desc.txt")
    args=parser.parse_args()
    regression( args.desc)