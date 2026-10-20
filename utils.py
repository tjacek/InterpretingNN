import json
import os,re

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    paths=[ path_i for id_i,path_i in iter_files(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def iter_files(path):
    if(type(path)==str):
         path=[path]
    for dir_i in path:
        for file_i in os.listdir(dir_i):
            yield  file_i,f'{dir_i}/{file_i}'

def filtered_files(path,taboo):
    if(type(taboo)==str):
        taboo=[taboo]
    if(type(taboo)==list):
        taboo=set(taboo)
    paths=[]
    for file_i in os.listdir(path):
        if(not file_i in taboo):
            paths.append(f'{path}/{file_i}')
    paths=sorted(paths,key=natural_keys)
    return paths    

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def find_paths(in_path,regex=r'^layer_\d+'):
    paths= [path_i 
                for id_i,path_i in iter_files(in_path)
                    if( re.match(regex,id_i))]
    paths=sorted(paths,key=natural_keys)
    return paths

def read_json(in_path):
    with open(in_path) as json_data:
        return SmartDict(json.load(json_data))

class DirProxy(object):
    def __init__(self,dir_path):
        self.path=dir_path
        make_dir(self.path)
        self.files={}
   
    def subpath(self,key):
        return f"{self.path}/{key}"

    def __getitem__(self,key):
        if(not key in self.files):
            path=self.subpath(key)
            make_dir(path)
            self.files[key]=path
        return self.files[key]

class SmartDict(dict):

    def split(self,pred):
        pos,neg={},{}
        for key_i,value_i in self.items():
            if(pred(key_i,value_i)):
                pos[key_i]=value_i
            else:
                neg[key_i]=value_i
        return SmartDict(pos),SmartDict(neg)

    def select(self,pred):
        if(not pred):
            return self
        if(type(pred)==list):
            pred=lambda key,value: key_i in pred
        new_dict={ key_i:value_i
                    for key_i,value_i in self.items()
                        if(pred(key_i,value_i))}
        return SmartDict(new_dict)

def bottom_files(in_path):
    bottom=[]
    for root, dir, files in os.walk(in_path):
        if(not dir):
            bottom.append(root)
    return bottom

def dir_paths(in_path,out_path):
    make_dir(out_path)
    for path_i in top_files(in_path):
        id_i=path_i.split("/")[-1]
        yield path_i,f"{out_path}/{id_i}"


def gini(arr):
    arr = np.sort(np.asarray(arr))
    n = len(arr)
    total = np.sum(arr)
    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    return np.sum((2*index - n - 1)*arr) / (n*total)

def slice_list(arr,step=10):
    n_iters=int(np.ceil(len(arr) / step))
    return [ arr[(i*step):(i+1)*step]   
                for i in range(n_iters)]

def dir_fun(fun):
    def helper(in_path,out_path):
        if(out_path):
            make_dir(out_path)
        for id_i,path_i in iter_files(in_path):
            if(out_path):
                out_i=f"{out_path}/{id_i}"
                fun(path_i,out_i)
            else:
                fun(path_i)
    return helper



def ineq(x,method="gini"):
    if(method=="gini"):
        return utils.gini(x)
    if(method=="CV"):
        return np.std(x)/np.mean(x)
    if(method=="L"):
        return np.amax(x)/np.sum(x)
    return 0