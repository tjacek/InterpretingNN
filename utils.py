import numpy as np
import os,re

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}' for file_i in os.listdir(path)]
    else:
        paths=path
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
    arr.sort()
    arr=np.array(arr)
    index = np.arange(1,arr.shape[0]+1)
    n = arr.shape[0]     
    return ((np.sum((2 * index - n  - 1) * arr)) / (n * np.sum(arr)))