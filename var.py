import numpy as np 
import base,dataset,deep
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import utils

class RandomSample(object):
    def __init__(self,bounds):
        self.bounds=bounds

    def __call__(self,n=1000):
        X=[]
        for max_i,min_i in self.bounds:
            x_i=np.random.uniform(max_i, min_i, n)
            X.append(x_i)
        return np.array(X).T
    
    def var_contr(self,model,out_path,n=1000):
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
        var_matrix=np.array(var_matrix)
        np.savetxt(f'{out_path}.txt', var_matrix, fmt='%f')
        heat_map(np.log(var_matrix))

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

def heat_map(var_matrix):
    import seaborn as sn
    sn.heatmap(var_matrix,cmap="YlGnBu",#linewidths=0.5,
        annot=True,annot_kws={"size": 5}, fmt='g')
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
        print(in_i)
        print(out_i)

if __name__ == '__main__':
    compute_var("uci","var_matrix")
#    random_exp("wine-quality-red",p=0.9)