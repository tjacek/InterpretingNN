import os.path
import base
import clfs
import dataset
import utils
#import deep

class ResultDir(utils.DirProxy):
    @property
    def splits(self):
        return self["splits"]
    
    @property
    def results(self):
        return self["results"]

class PredDict(object):
    def __init__(self,df_dict):
        self.df_dict=df_dict

    def __getitem__(self,clf):
        clf_dict={}
        for name_i,df_i in self.df_dict.items():
            acc = df_i.set_index("clf")["norm_acc"]
            clf_dict[name_i]=acc[clf]
        return clf_dict

    def issubset(self, names):
        return set(names).issubset(self.df_dict.keys())

def make_pred( data_path,
               dir_path,
               clf_types=None):
    if(clf_types is None):
        clf_types=list(clfs.TYPES.keys())
    dir_proxy=ResultDir(dir_path)
    for id_i,path_i in utils.iter_files(data_path):
        data_i=dataset.read_csv(path_i)
        split_path_i=f"{dir_proxy.splits}/{id_i}"
        splits_i=get_splits(split_path_i,data_i)
        result_path_i=f"{dir_proxy.results}/{id_i}"
        utils.make_dir(result_path_i)
        for type_j in clf_types:
            print(type_j)
            results,_=splits_i(data_i,type_j)
            results.save(f"{result_path_i}/{type_j}")
            print(results.acc)

def get_splits(split_path,data):
    split_cls=base.SplitGroup
    if os.path.exists(split_path):
        return split_cls.read(split_path)
    else:
        splits= split_cls.make( data,
                                n_repeats=1,
                                n_splits=10)
        splits.save(split_path)
    return splits

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
    return PredDict(df_dict)

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
    make_pred("uci/s_data","uci")
#    show_pred(["AutoML/output","uci/output"])