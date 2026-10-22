from dataclasses import dataclass
import os.path
import base
import clfs
import dataset
import utils

@dataclass
class ExpParams:
    data_path:str      
    split_path:str
    out_path:str = None
    clf_type:str = 'RF' 
    k:int = 100 
    
    @property
    def id(self):
        return self.data_path.split("/")[-1]    

    def get_data(self):
        data=dataset.read_csv(self.data_path)
        splits=base.SplitGroup.read(self.split_path)
        return data,splits

    def get_clf(self):
        return clfs.TYPES[self.clf_type]

    def attr_iter(self,attr,values):
        for value_i in values:
            exp_i = ExpParams(**self.__dict__)
            setattr(exp_i, attr, value_i)
            yield exp_i
    
    def data_iter(self):
        if(not os.path.isdir(self.data_path)):
            yield self
            return
        utils.make_dir(self.out_path)
        for id_i,data_i in utils.iter_files(self.data_path):
            split_i=f"{self.split_path}/{id_i}"
            exp_i=ExpParams(**self.__dict__)
            exp_i.data_path=data_i
            exp_i.split_path+=id_i
            exp_i.out_path+=id_i
            yield exp_i

def from_conf( conf_dict):
    multi,single=conf_dict.split(lambda k,v: type(v)==list)
    exp=ExpParams(**single)
    base_exps=exp.data_iter()
    all_exps=[]
    for exp_i in base_exps:
        utils.make_dir(exp_i.out_path)
        all_exps+=from_dict(exp_i,multi)
    return all_exps

def from_dict(prototype,arg_dict):
    exps=[prototype]
    for key_i,value_i in arg_dict.items():
        new_exps=[]
        for exp_j in exps:
            utils.make_dir(exp_j.out_path)
            exps_j=list(exp_j.attr_iter(key_i,value_i))
            for exp_k in exps_j:
                value_k=getattr(exp_k, key_i)
                exp_k.out_path+=f"/{value_k}"
                new_exps.append(exp_k)
        exps=new_exps
    return exps