import pandas as pd
import base,clf,dataset,deep,utils

def basic_exp(in_path):
    clf_types= clf.Clf.to_dict()
    print(clf_types)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        splits_i=base.SplitGroup.make( data_i,
                                       n_repeats=1,
                                       n_splits=10)
        for name_j,clf_type_j in clf_types.items():
            print(name_j)
            results=splits_i(data_i,clf_type_j)
            print(results.get_acc())

def make_models(in_path,out_path):
    clf_types=[clf.RF,clf.GRAD,clf.LR,clf.SVM]
    utils.make_dir(out_path)
    for id_i,path_i in utils.iter_files(in_path):
        print(path_i)
        data_i=dataset.read_csv(path_i)
        splits_i=base.SplitGroup.make( data_i,
                                       n_repeats=1,
                                       n_splits=10)
        out_i=f"{out_path}/{id_i}"
        utils.make_dir(out_i)
        splits_i.save(f"{out_i}/splits")
        for type_j in clf_types:
            out_ij=f"{out_i}/{type_j.NAME}"
            utils.make_dir(out_ij)
            results=splits_i(data_i,type_j)
            results.save(f"{out_ij}/results")
            print(results.get_acc())

def show_pred(in_path):
    reader=base.ResultGroup.read
    for id_i,path_i in utils.iter_files(in_path):
        paths=utils.filtered_files(path_i,"splits")
        def helper(path_i):
            clf_i=path_i.split("/")[-1]
            result=reader(f"{path_i}/results")
            return [id_i,clf_i,result.get_acc()]
        df=dataset.make_df(helper,
                           iterable=paths,
                           cols=["data","clf","acc"])
        acc=df["acc"].tolist()
        min_acc=min(acc)
        delta_acc= max(acc)-min_acc
        df["norm_acc"]=df["acc"].apply(lambda acc: (acc-min_acc)/delta_acc)
        print(df)

if __name__ == '__main__':
    show_pred("output")