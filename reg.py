import dataset,train

def input_data(data_path,result_path):
    df=dataset.make_desc(data_path)
    df_dict=train.show_pred(result_path,verbose=False)
    def helper(row):
        data_id=row["data"]
        df_i=df_dict[data_id]
        acc = df_i.set_index("clf")["norm_acc"]
        return acc["RF"] - acc["MLP"]
    df["diff"]=df.apply(helper, axis=1)
    print(df)

input_data("spatial","output")