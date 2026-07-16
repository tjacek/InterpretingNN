from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from dataclasses import dataclass,field#asdict
import dataset,train,plot,utils

@dataclass
class RegOutput:
    names: list = field(default_factory=list)
    y_true: list = field(default_factory=list)
    y_pred: list = field(default_factory=list)
    
    def __len__(self):
        return len(self.names)

    def add( self,
             name_i,
             true_i,
             pred_i):
        self.names.append(name_i)
        self.y_true.append(true_i)
        self.y_pred.append(pred_i)        
    
    def raw_error(self):
        return np.array(self.y_true) - np.array(self.y_pred)

    def abs_error(self):
        abs_error=np.abs(self.raw_error())
        return np.mean(abs_error)

    def mse(self):
        error=self.raw_error()
        return np.sqrt(np.mean(error**2))
    
    def slice(self,i,step=10):

        arr={ key_i:self.__dict__[key_i][i*step:(i+1)*step] 
                for key_i in self.__dict__}
        return self.__class__(**arr)

    def iter_slices(self,step=10):
        n_iters=int(np.ceil(len(self) / step))
        for i in range(n_iters):
            yield self.slice(i,step)
    
    @classmethod
    def make(cls,df):
        output=cls()
        for train_i,test_i,data_i  in leve_one_out(df):
            pred_i=output.fit(train_i,test_i)
            output.add(data_i,test_i.y,pred_i)
        return output

def leve_one_out(df,norm=False):
    for i in range(len(df)):
        train = df.drop(index=i)
        X_train = train.drop(columns=["target", "data"]).to_numpy()
        y_train = train["target"].to_numpy()
        train=dataset.Dataset(X_train,y_train)

        X_test = df.iloc[[i]].drop(columns=["target", "data"]).to_numpy()
        y_test = df.iloc[i]["target"]
        test=dataset.Dataset(X_test,y_test)

        data_i = df.iloc[i]["data"]
        if(norm):
            train.norm()
            test.norm()
        yield train,test,data_i 

@dataclass
class GaussOutput(RegOutput):
    error: float  = field(default_factory=list)
    
    def fit(self,train_i,test_i):
        kernel =  RBF(length_scale=1.0, 
                      length_scale_bounds=(1e-2, 1e2))
        gauss_process = GaussianProcessRegressor(kernel=kernel, 
                                                    n_restarts_optimizer=9)
        gauss_process.fit(train_i.X, train_i.y)
        mean_i, std_i = gauss_process.predict(test_i.X, 
                                          return_std=True)
        self.error.append(std_i[0])
        return mean_i[0]
    
    def show(self,df):
        def img_iter():
            for out_i in self.iter_slices(10):
                yield plot.error_hist(**out_i.__dict__)
        return img_iter()

@dataclass
class LinearOutput(RegOutput):
    coef: float  = field(default_factory=list)

    def fit(self,train_i,test_i):
        reg_i = LinearRegression()
        reg_i.fit(train_i.X,train_i.y)
        mean_i= reg_i.predict(test_i.X)
        self.coef.append(reg_i.coef_)
        return mean_i

    def show(self,df):
        coef_arr=np.array(self.coef)
        value=np.mean(coef_arr,axis=0)
        var=np.std(coef_arr,axis=0)
        df=df.drop(["data","target"],axis=1)
        cols=df.columns
        for i,col_i in enumerate(cols):
            print(f"{col_i}:{value[i]:.4f},{var[i]:.4f}")

def regression( df_path,
                result_path,
                reg_alg="gauss",
                out_path=None):
    df=get_input_data(df_path,
                      result_path,
                      x_clf="RF",
                      y_clf="TabPFN")
    if(reg_alg=="gauss"):
        reg_alg=GaussOutput
    else:
        reg_alg=LinearOutput
    output=reg_alg.make(df)
    print(f"Mean absolute error:{output.abs_error():.4f}")
    print(f"Mean squared error {output.mse():.4f}")
    img_iter=output.show(df)
    if(img_iter):
        plot.show_plots(img_iter,out_path)

def get_input_data(df_path,
                   result_path,
                   x_clf="RF",
                   y_clf="TabPFN"):
    df=pd.read_csv(df_path)
    df_dict=train.show_pred(result_path,verbose=False)
    x_dict=df_dict[x_clf]
    y_dict=df_dict[y_clf]
    def helper(row):
        data_id=row["data"]
        return x_dict[data_id] - y_dict[data_id]
    df["target"]=df.apply(helper, axis=1)
    df=df.sort_values(by="target")
    print(df)
    return df

def to_array(df):
    df=df.drop("data",axis=1)
    y=df["target"].to_numpy()
    df=df.drop("target",axis=1)
    X=df.to_numpy()
    return X,y

regression( "desc/full",
            ["AutoML/output",
             "uci/output"],
           "gauss",None)
#           "gauss_reg")