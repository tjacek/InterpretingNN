import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import argparse
import dataset,plot,shape,utils

class GruopOfFeature(object):
    def __init__(self,features):
        self.features=features

    def names(self):
        names=["data"]
        for feature_i in self.features:
            names_i= feature_i.names()   
            if(not isinstance(names_i, list)):
                names_i=[names_i]
            names.extend(names_i)
        return names

    def __call__(self,arg_dict):
        values=[arg_dict["id"]]
        for feature_i in self.features:
            values_i= feature_i(arg_dict)   
            if(not isinstance(values_i, list)):
                values_i=[values_i]
            values.extend(values_i)
        return values

class Feature(ABC):
    def names(self):
        return str(self)
   
    @abstractmethod
    def __call__(self,data):
        pass

class Basic(Feature):
    def names(self):
        return ["classes","samples", "feats" ]

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        data_params=data.params
        return data_params.to_list()

class PcaFeats(Feature):
    def names(self):
        return ["pca_max","pca_95"]

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        pca = PCA(n_components=None)
        pca.fit(data.X)
        exp_var=pca.explained_variance_ratio_
        greatest=exp_var[0]
        cum_var=np.cumsum(exp_var)
        int_var=(cum_var>0.95).astype(int)
        thres_var= np.argmax(int_var)/cum_var.shape[0]
        return [greatest,thres_var]

class IR(Feature):
    def __init__( self,log=False):
        self.log=log

    def __str__(self):
        return "IR"

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        data_params=data.params
        sizes=data_params.sizes()
        ir= max(sizes)/min(sizes)
        if(self.log):
            return np.log(ir)
        return ir

class GINI(Feature):
    def __str__(self):
        return "gini"

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        data_params=data.params()
        return utils.gini(data_params.sizes())

class Shapley(Feature):
    def __str__(self):
        return "shapley"

    def __call__(self,arg_dict):
        values=arg_dict["shapley"]
        values=np.abs(values)
        values/=np.sum(values,axis=0)
        values=np.amax(values,axis=0)
        return utils.gini(values)

class Corel(Feature):
    def __str__(self):
        return "corel"
    
    def __call__(self,arg_dict):
        shapley=arg_dict["shapley"]
        ablat=arg_dict["ablat"]
        x=shapley.flatten()
        y=ablat.flatten()
        r, p = pearsonr(x, y)
        if(np.isnan(r)):
            r=0.0
        return r

class Infl(Feature):
    def __init__( self,
                  infl_type="ablat",
                  scale=True):
        self.infl_type=infl_type
        self.scale=scale
    
    def __str__(self):
        return f"Infl({self.infl_type})"

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        infl=arg_dict[self.infl_type]

        params=data.params
        if(params.cats<3):
            return 0.0
        min_index=params.min_cls()
        infl_i= infl[min_index,:]
        print(arg_dict["id"])
        print(infl_i)
        value = utils.gini(np.abs(infl_i))
        if(self.scale):
            min_size=params.sizes_dict[min_index]
            prop=  params.avrage_size()/min_size
            return value*prop
        return value

class FeatInfl(Feature):
    def __init__( self,
                  infl_type="ablat"):
        self.infl_type=infl_type

    def __str__(self):
        return f"FeatInfl({self.infl_type})"   
     
    def __call__(self,arg_dict):
        data=arg_dict["data"]
        infl=arg_dict[self.infl_type]
        if(data.params.cats==2):
            return 0
        min_index=data.params.min_cls()
        cls_min=infl[min_index,:]
        max_index=np.argmax(cls_min)
        feat_max=infl[:,max_index]
        print(feat_max.shape)
        
        min_size=data.params.sizes_dict[min_index]
        prop=  np.log(data.params.samples)
        prop/=min_size
        return prop*utils.gini(np.abs(feat_max))

class CorlSize(Feature):
    def __str__(self):
        return f"CorlSize"

    def __call__(self,arg_dict):
        data=arg_dict["data"]
        infl=arg_dict["ablat"]
        params=data.params
        if(params.cats==2):
            return 0
        x= [ params.sizes_dict[i]/params.samples 
                  for i in range(params.cats)]
        y=[utils.ineq(np.abs(x_i),"L") for x_i in infl]

        print(y)
        r,p=plot.plot_xy( x,y,
                   title=arg_dict["id"], 
                   x_label="size",
                   y_label="gini")
        return np.amax(x)

def make_desc( conf_path,
               features_list=None,
               order_by="feats",
               out_path=None):
    conf_dict=utils.read_json(conf_path)
    args_dict=get_arg_dicts(conf_dict["data"],
                             conf_dict["sources"])
    if(features_list is None):
        features_list=[Basic(),PcaFeats(),IR(),GINI()]
    features= GruopOfFeature(features_list)
    df=dataset.make_df( helper=features,
                        iterable=args_dict,
                        cols=features.names(),
                        multi=False)
    df.sort_values(by=order_by,inplace=True)
    print(df.round(4))
    if(out_path):
        df.to_csv(out_path, sep=",", index=False)

def get_arg_dicts(data_path,sources):
    data=dataset.get_data_dict(data_path)
    source_dicts={"data":data} 
    for type_i,path_i in sources:
        source_dicts[type_i]=plot.get_matrix_dict(path_i)
    arg_dicts=[]
    for id_i in data:
        arg_i={"id":id_i}
        for key_i,source_i in source_dicts.items():
            arg_i[key_i]=source_i[id_i]
        arg_dicts.append(arg_i)
    return arg_dicts


def plot_xy(x_path,y_path):
    x_dict=plot.get_matrix_dict(x_path)
    y_dict=plot.get_matrix_dict(y_path)
    for key_i in x_dict:
        x_i=x_dict[key_i].flatten()
        y_i=y_dict[key_i].flatten()
        plot.plot_xy( x_i,y_i,
                      title=key_i, 
                      x_label=x_path,
                      y_label=y_path)

def transpose(in_path,out_path):
    utils.make_dir(out_path)
    mat_dict=plot.get_matrix_dict(in_path)
    for name_i,mat_i in mat_dict.items():
        np.savetxt(f'{out_path}/{name_i}', mat_i.T, fmt='%f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str,default="matrix/conf.json")
    parser.add_argument("--order", type=str,default="Infl(shapley)")
    parser.add_argument("--out", type=str,default=None)#"desc/infl3")
#    features_list=[Basic(),Infl("ablat"),Infl("shapley")]#,IR(True)]
    features_list=[Basic(),#FeatInfl("ablat"),FeatInfl("shapley"),
                    Infl("ablat"),Infl("shapley")]
#    features_list=[Basic(),CorlSize()]
    args=parser.parse_args()
    make_desc( args.conf,
               features_list=features_list,
               order_by=args.order,
               out_path=args.out)
#    transpose("matrix/shapley","matrix/shapley_")
