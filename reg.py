from sklearn.linear_model import LinearRegression
import dataset,train

def get_input_data(data_path,result_path):
    df=dataset.make_desc(data_path)
    df_dict=train.show_pred(result_path,verbose=False)
    def helper(row):
        data_id=row["data"]
        df_i=df_dict[data_id]
        acc = df_i.set_index("clf")["norm_acc"]
        return acc["RF"] - acc["MLP"]
    df["target"]=df.apply(helper, axis=1)
    return df

def to_array(df):
    df=df.drop("data",axis=1)
    y=df["target"].to_numpy()
    df=df.drop("target",axis=1)
    X=df.to_numpy()
    return X,y

def reg_exp(data_path,result_path):
    df=get_input_data(data_path,result_path)
    X,y=to_array(df)
    reg = LinearRegression().fit(X, y)
    reg.fit(X,y)
    print(f"R:{reg.score(X, y)}")
    cols=df.columns[1:-1]
    for name_i,coef_i in zip(cols,reg.coef_):
        print(f"{name_i}:{coef_i:.4f}")
    y_pred=reg.predict(X)
    for i,pred_i in enumerate(y_pred):
        row_i=df.iloc[i]
        data_i,y_true=row_i["data"],row_i["target"]
        print(f"{data_i},{y_true:.4f},{pred_i:.4f},{pred_i-y_true:.4f}")

reg_exp("spatial","output")