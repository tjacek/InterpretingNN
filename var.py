import numpy as np 
import base,dataset,deep
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import utils

class VarMatrix(object):
    def __init__(self,arr):
        self.arr=arr

    def __call__(self,fun,mode="cats"):
        if(mode=="cats"):
            return [fun(arr_i) for arr_i in self.arr.T]
        elif(mode=="feats"):
            return [fun(arr_i) for arr_i in self.arr]
        else:
            return fun(self.arr.flatten())
class RandomSample(object):
    def __init__(self,bounds):
        self.bounds=bounds

    def __call__(self,n=1000):
        X=[]
        for max_i,min_i in self.bounds:
            x_i=np.random.uniform(max_i, min_i, n)
            X.append(x_i)
        return np.array(X).T
    
    def var_contr(self,model,n=1000):
        x=self(n)
        y=model.predict_proba(x)
        def helper(dim):
            all_sqr=[]
            for i,x_i in enumerate(tqdm(x)):
                delta_x_i=self.change(x_i,dim,n)
                delta_y_i=model.predict_proba(delta_x_i)
                y_i=y[i]
                for y_j in delta_y_i:
                    err=y_i-y_j
                    sqr_err=err**2
                    all_sqr.append(sqr_err)
            return np.mean(all_sqr,axis=0)
        n_dims=x.shape[1]
        print(f"Dims:{n_dims}")
        var_matrix=[]
        for i in range(n_dims):
            var_i=helper(i)
            var_matrix.append(var_i)
            print(var_i)
        return np.array(var_matrix)


    def change(self,x,cord_i,n):
        max_i,min_i=self.bounds[cord_i]
        new_x=[]
        for _ in range(n):
            x_j=x.copy()
            x_j[cord_i]=np.random.uniform(max_i,min_i)
            new_x.append(x_j)
        return np.array(new_x)

    def iter_sampling( self,
                        model,
                        n_iters=10,
                        n_samples=1000):
        exp,var=[],[]
        for i in range(n_iters):
            x_i=self(n_samples)
            y_prob=model.predict_proba(x_i)
            exp_i=np.mean(y_prob,axis=0)
            var_i=np.std(y_prob,axis=0)
            exp.append(exp_i)
            var.append(var_i)
        return np.array(exp),np.array(var)

def heat_map(var_matrix,name=""):
    import seaborn as sn
    sn.heatmap(var_matrix,cmap="YlGnBu",#linewidths=0.5,
        annot=True,annot_kws={"size": 5}, fmt='g')
    plt.title(name)
    plt.show()

def random_exp(in_path,p):
    data=dataset.read_csv(in_path)
    split=base.random_split(len(data),p=p)
    sampler=RandomSample(data.range())
    nn=deep.single_builder(params=data.params_dict())
    split=base.random_split(len(data),p=0.9)
    result,_=split.eval(data,nn)
    sampler.var_contr(nn,in_path)

def compute_var(in_path,out_path):
    for in_i,out_i in utils.dir_paths(in_path,out_path):
        data_i=dataset.read_csv(in_i)
        split_i=base.random_split(len(data_i),p=0.9)
        nn_i=deep.single_builder(params=data_i.params_dict())
        result,_=split_i.eval(data_i,nn_i)
        sampler_i=RandomSample(data_i.range())
        print(in_i)
        var_i=sampler_i.var_contr(nn_i)
        np.savetxt(out_i, var_i, fmt='%f')
#        heat_map(np.log(var_matrix))

def read_var(in_path):
    for in_i in utils.top_files(in_path):
        var_i = np.loadtxt(in_i)
        id_i=in_i.split("/")[-1]
        yield id_i,VarMatrix(var_i)

def show_heatmap(in_path,norm=False):
    for id_i,var_i in read_var(in_path):
        if(norm):
            var_i=[ var_j/np.sum(var_j) 
                    for var_j in var_i.T]
            var_i=np.array(var_i).T
        heat_map(var_i,id_i)

def show_gini(in_path,mode="cats"):
    for id_i,var_i in read_var(in_path):
        print(id_i)
        print(var_i(utils.gini,mode))

def show_stats(in_path,mode=None):
    stat_fun={ "max_fun":lambda x: np.amax(np.log(x)),
               "mean_fun":lambda x: np.mean(np.log(x)),
               "min_fun":lambda x: np.amin(np.log(x))
             }
    for id_i,var_i in read_var(in_path):
        print(id_i)
        for name_j,fun_j in stat_fun.items():
            print(f"{name_j}:{var_i(fun_j,mode)}")

if __name__ == '__main__':
#    compute_var("uci","var_matrix")
    show_stats("var_matrix",None)
#    random_exp("wine-quality-red",p=0.9)