from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,plot_tree,DecisionTreeClassifier
#import sklearn.tree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass,field#asdict
import matplotlib.pyplot as plt
import dataset,train,plot,utils

@dataclass
class LearningAlg:
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

class RegAlg(LearningAlg):
    def raw_error(self):
        return np.array(self.y_true) - np.array(self.y_pred)

    def abs_error(self):
        abs_error=np.abs(self.raw_error())
        return np.mean(abs_error)

    def mse(self):
        error=self.raw_error()
        return np.sqrt(np.mean(error**2))

    def print_err(self):
        print(f"Mean absolute error:{self.abs_error():.4f}")
        print(f"Mean squared error {self.mse():.4f}")


class ClfAlg(LearningAlg):
    def print_err(self):
        acc=accuracy_score(self.y_true,self.y_pred)
        f1=f1_score(self.y_true,self.y_pred)
        print(f"Accuracy:{acc:.4f}")
        print(f"F1-score {f1:.4f}")
    
    def error_dict(self):
        pairs= zip(self.y_true,self.y_pred)
        cat_dict={i:[]  for i in list(set(self.y_true))}
        for i,(true_i,pred_i) in enumerate(pairs):
            if(true_i!=pred_i):
                cat_dict[true_i].append(self.names[i])
        return cat_dict

def leve_one_out(df,norm=True):
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        train_df = df.drop(i)
        test_df = df.iloc[[i]]

        X_train = train_df.drop(columns=["target", "data"]).to_numpy()
        y_train = train_df["target"].to_numpy()
        
        X_test = test_df.drop(columns=["target", "data"]).to_numpy()
        y_test = test_df["target"].iloc[0]
        
        train = dataset.Dataset(X_train, y_train)
        test = dataset.Dataset(X_test, y_test)

        data_i = test_df["data"].iloc[0]
        if(norm):
            scaler = StandardScaler()
            train.X = scaler.fit_transform(train.X)
            test.X = scaler.transform(test.X)
#            mean_y,std_y=np.mean(train.X),np.std(train.y)  
#            train.y = (train.y-mean_y)/std_y
#            test.y = (test.y-mean_y)/std_y
        yield train,test,data_i 

def get_cols(df):
    df=df.drop(["data","target"],axis=1)
    return df.columns

@dataclass
class GaussAlg(RegAlg):
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
class LinearAlg(RegAlg):
    coef: float  = field(default_factory=list)

    def fit(self,train_i,test_i):
        reg_i = LinearRegression()
        reg_i.fit(train_i.X,train_i.y)
        mean_i= reg_i.predict(test_i.X)
        self.coef.append(reg_i.coef_)
        return mean_i

    def show(self,df):
        print_coeff(self.coef,get_cols(df))

def print_coeff(coef,names):
    coef_arr=np.array(coef)
    value=np.mean(coef_arr,axis=0)
    var=np.std(coef_arr,axis=0)
    for i,col_i in enumerate(names):
        print(f"{col_i}:{value[i]:.4f},{var[i]:.4f}")

class RegTreeAlg(RegAlg):
    def __init__(self):
        super().__init__()
        self.reg=None

    def fit(self,train_i,test_i):
        self.reg = DecisionTreeRegressor(max_depth=4, 
                                          random_state=42)
        self.reg.fit(train_i.X,train_i.y)
        pred= self.reg.predict(test_i.X)
        return pred

    def show(self,df):   
        plt.figure(figsize=(20, 10))
        plot_tree(
                  self.reg,
                  feature_names=get_cols(df),
                  filled=True,
                  rounded=True,
                  fontsize=10
               )
        plt.title("Decision Tree Structure")
        plt.show()

class TreeAlg(ClfAlg):
    def __init__(self):
        super().__init__()
        self.reg=None

    def fit(self,train_i,test_i):
        self.reg = DecisionTreeClassifier(max_depth=4, 
                                          random_state=42)
        self.reg.fit(train_i.X,train_i.y)
        pred= self.reg.predict(test_i.X)
        return pred

    def show(self,df):
        print(self.error_dict())
        plt.figure(figsize=(20, 10))
        plot_tree(
                  self.reg,
                  feature_names=get_cols(df),
                  filled=True,
                  rounded=True,
                  fontsize=10
               )
        plt.title("Decision Tree Structure")
        plt.show()

@dataclass
class LogisticAlg(ClfAlg):
    coef: float  = field(default_factory=list)

    def fit(self,train_i,test_i):
        reg_i = LogisticRegression(solver='liblinear')
        train_i.y=train_i.y.astype(int)
        reg_i.fit(train_i.X,train_i.y)
        mean_i= reg_i.predict(test_i.X)
        self.coef.append(reg_i.coef_.flatten())
        return mean_i[0]

    def show(self,df):
        print_coeff(self.coef,get_cols(df))        
        print(self.error_dict())

def regression( df_path,
                result_path,
                alg_type="gauss",
                out_path=None):
    reg_alg=get_reg_alg(alg_type)
    df=get_input_data(df_path,
                      result_path,
                      x_clf="RF",
                      y_clf="TabPFN",
                      discreet=isinstance(ClfAlg,reg_alg))
    output=reg_alg.make(df)
    output.print_err()
    img_iter=output.show(df)
    if(img_iter):
        plot.show_plots(img_iter,out_path)

def get_reg_alg(alg_type):
    if(alg_type=="gauss"):
        return GaussAlg
    elif(alg_type=="tree"):
        return TreeAlg
    elif(alg_type=="logistic"):
        return LogisticAlg
    else:
        return LinearAlg

def get_input_data(df_path,
                   result_path,
                   x_clf="RF",
                   y_clf="TabPFN",
                   discreet=False):
    df=pd.read_csv(df_path)
    df_dict=train.show_pred(result_path,verbose=False)
    x_dict=df_dict[x_clf]
    y_dict=df_dict[y_clf]
    def helper(row):
        data_id=row["data"]
        return x_dict[data_id] - y_dict[data_id]
    df["target"]=df.apply(helper, axis=1)
    if(discreet):
        df["target"]=df.apply(lambda x:int(x["target"]>0), axis=1)
    df=df.sort_values(by="target")
    print(df)
    return df

def to_array(df):
    df=df.drop("data",axis=1)
    y=df["target"].to_numpy()
    df=df.drop("target",axis=1)
    X=df.to_numpy()
    return X,y

regression( "desc/ineq",
            ["AutoML/output",
             "uci/output"],
           "gauss",None)
#           "gauss_reg")