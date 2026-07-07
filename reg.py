from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import dataset,train

def get_input_data(data_path,
                   result_path,
                   x_clf="RF",
                   y_clf="TabPFN"):
    df=dataset.make_desc(data_path)
    df_dict=train.show_pred(result_path,verbose=False)
    x_dict=df_dict[x_clf]
    y_dict=df_dict[y_clf]
    def helper(row):
        data_id=row["data"]
        return x_dict[data_id] - y_dict[data_id]
    df["target"]=df.apply(helper, axis=1)
    return df

def to_array(df):
    df=df.drop("data",axis=1)
    y=df["target"].to_numpy()
    df=df.drop("target",axis=1)
    X=df.to_numpy()
    return X,y

def leve_one_out(df):
    for i in range(len(df)):
        train = df.drop(index=i)

        X_train = train.drop(columns=["target", "data"]).to_numpy()
        y_train = train["target"].to_numpy()
        train=dataset.Dataset(X_train,y_train)

        X_test = df.iloc[[i]].drop(columns=["target", "data"]).to_numpy()
        y_test = df.iloc[i]["target"]
        test=dataset.Dataset(X_test,y_test)

        data_i = df.iloc[i]["data"]

        yield train,test, data_i

def gpr(df):
    names,y_true,y_pred,error=[],[],[],[]
    for train_i,test_i,data_i  in leve_one_out(df):
        kernel =  RBF(length_scale=1.0, 
                      length_scale_bounds=(1e-2, 1e2))
        gauss_process = GaussianProcessRegressor(kernel=kernel, 
                                                    n_restarts_optimizer=9)
        gauss_process.fit(train_i.X, train_i.y)
        names.append(data_i)
    
        mean_i, std_i = gauss_process.predict(test_i.X, 
                                              return_std=True)
        y_pred.append(mean_i[0])
        error.append(std_i[0])
    return names,y_true,y_pred,error

def gauss_reg(data_path,result_path):
    df=get_input_data(data_path,result_path)
    names,y_true,y_pred,error=gpr(df)

#def reg_exp(data_path,result_path):
#    df=get_input_data(data_path,result_path)
#    X,y=to_array(df)
#    reg = LinearRegression().fit(X, y)
#    reg.fit(X,y)
#    print(f"R:{reg.score(X, y)}")
#    cols=df.columns[1:-1]
#    for name_i,coef_i in zip(cols,reg.coef_):
#        print(f"{name_i}:{coef_i:.4f}")
#    y_pred=reg.predict(X)
#    def helper(arg):
#        i,pred_i=arg
#        row_i=df.iloc[i]
#        data_i,y_i=row_i["data"],row_i["target"]
#        return [data_i,y_i,pred_i, y_i-pred_i]
#    df_reg=dataset.make_df(helper,
#                           enumerate(y_pred),
#                           ["data","true","pred","res"] )
#    print(df_reg)

gauss_reg(["AutoML/data",
           "uci/data"],
          ["AutoML/output",
           "uci/output"])