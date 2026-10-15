from dataclasses import dataclass
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

    def iter_exp(self,attr,values):
        for value_i in values:
            exp_i = ExpParams(**self.__dict__)
            setattr(exp_i, attr, value_i)
            yield exp_i

def from_dir( data_path,
              split_path,
              clfs):
    exps=[]
    for id_i,data_i in utils.iter_files(data_path):
        split_i=f"{split_path}/{id_i}"
        exp_i=ExpParams( data_i,
                         split_i,
                         clfs)
        exps.append(exp_i)
    return exps

def from_conf( conf_dict):
    multi,single=conf_dict.split(lambda k,v: type(v)==list)
    exps=[ExpParams(**single)]
    utils.make_dir(exps[0].out_path)
    for key_i,value_i in multi.items():
        new_exps=[]
        for exp_j in exps:
            utils.make_dir(exp_j.out_path)
            exps_j=list(exp_j.iter_exp(key_i,value_i))
            for exp_k in exps_j:
                value_k=getattr(exp_k, key_i)
                exp_k.out_path+=f"/{value_k}"
                new_exps.append(exp_k)
        exps=new_exps
    return exps