import pandas as pd
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

@dataclass
class LinearAlg(object):
    coef: float  = field(default_factory=list)

    def fit(self,train_i,test_i):
        reg_i = LinearRegression()
        reg_i.fit(train_i.X,train_i.y)
        mean_i= reg_i.predict(test_i.X)
        self.coef.append(reg_i.coef_)
        return mean_i

def regression(desc_path):
    df=pd.read_csv(desc_path)
    desc=DescDataset(df)
    print(desc.feats())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg_type", type=str,default="logistic")
    parser.add_argument("--desc", type=str,default="uci/desc.txt")
    args=parser.parse_args()
    regression( args.desc)