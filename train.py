import os.path
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

def make_pred(in_path,out_path):
#    clf_types=[clf.RF,clf.GRAD,clf.LR,clf.SVM]
    clf_types=[deep.TabPFN]
    utils.make_dir(out_path)
    for id_i,path_i in utils.iter_files(in_path):
        print(path_i)
        out_i=f"{out_path}/{id_i}"
        utils.make_dir(out_i)
        data_i=dataset.read_csv(path_i)
        splits_i=get_splits(out_i,data_i)
        for type_j in clf_types:
            out_ij=f"{out_i}/{type_j.NAME}"
            if(os.path.exists(out_ij)):
                continue
            utils.make_dir(out_ij)
            results,_=splits_i(data_i,type_j)
            results.save(f"{out_ij}/results")
            print(results.get_acc())

def show_pred(in_path,verbose=True):
    reader=base.ResultGroup.read
    df_dict={}
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
        df.sort_values(by="norm_acc",inplace=True)
        if(verbose):
            print(df)
        df_dict[id_i]=df
    return df_dict

def get_splits(in_path,data_i):
    split_cls=base.SplitGroup
    split_path=f"{in_path}/splits"
    print(split_path)
    if os.path.exists(split_path):
        return split_cls.read(split_path)
    else:
        splits= split_cls.make( data_i,
                               n_repeats=1,
                               n_splits=10)
        splits.save(split_path)
    return splits

def make_models(in_path,out_path):
    clf_type=deep.MLP
    for id_i,path_i in utils.iter_files(in_path):
        out_i=f"{out_path}/{id_i}"
        print(id_i)
        data_i=dataset.read_csv(path_i)
        splits_i=get_splits(out_i,data_i)
        clf_path_i=f"{out_i}/{clf_type.NAME}"
        utils.make_dir(clf_path_i)
        results,clfs=splits_i(data_i,clf_type)
        results.save(f"{clf_path_i}/results")
        utils.make_dir(f"{clf_path_i}/models")
        for j,clf_j in enumerate(clfs):
            clf_j.save(f"{clf_path_i}/models/{j}")
        print(results.get_acc())

if __name__ == '__main__':
    make_pred("AutoML/data","AutoML/output")
#    make_models("AutoML/data","AutoML/output")
    show_pred(["AutoML/output","uci/output"])