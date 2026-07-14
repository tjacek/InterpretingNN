import numpy as np
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
