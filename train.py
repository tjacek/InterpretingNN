import os.path
import base
import clfs
import dataset
import utils
import argparse

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

    def diff(self,clf_x,clf_y):
        x=self[clf_x]
        y=self[clf_y]
        return { data_i:x[data_i]-y[data_i] for data_i in x}

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

def get_pred(dir_path,verbose=True):
    reader=base.ResultGroup.read
    dir=ResultDir(dir_path)
    df_dict={}
    for id_i,path_i in utils.iter_files(dir.results):
        def helper(pair):
            id_j,path_j=pair
            result_j=reader(path_j)
            return [id_i,id_j,result_j.acc]
        df_i=dataset.make_df( helper=helper,
                              iterable=utils.iter_files(path_i),
                              cols=["data","clf","acc"])
        acc=df_i["acc"].tolist()
        min_acc=min(acc)
        delta_acc= max(acc)-min_acc
        df_i["norm_acc"]=df_i["acc"].apply(lambda acc: (acc-min_acc)/delta_acc)
        df_i.sort_values(by="norm_acc",inplace=True)
        if(verbose):
            print(df_i)
        df_dict[id_i]=df_i
    return PredDict(df_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="uci/s_data",)
    parser.add_argument("--dir", type=str, default="uci")
    parser.add_argument("--cmd", type=str, default="show")
    args=parser.parse_args()
    if(args.cmd=="make"):
        make_pred(args.data,args.dir)
    if(args.cmd=="show"):
        pred_dict=get_pred(args.dir)
        print(pred_dict.diff("RF","GRAD"))
