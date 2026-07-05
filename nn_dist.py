import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import deep, utils

def neural_dist( in_path,
	             norm_fun,
	             out_path=None):
	if(out_path):
		utils.make_dir(out_path)

	paths=mlp_paths(in_path)
	for id_i,path_i in paths.items():
		mlp_i=deep.MLP.read(f"{path_i}/models/0.keras")
		weights=mlp_i.get_weights()
		x,y_dict=compute_density(weights,norm_fun)
		out_i= f"{out_path}/{id_i}" if(out_path) else None
		plot(x,y_dict,id_i,out_i)

def plot( x,
	      y_dict,
	      title,
	      out_path=None):
	plt.figure(figsize=(8, 5))
	for id_i,y_i in y_dict.items():
		plt.plot(
            x[:, 0],
            y_i,
            label=id_i
        )
	plt.xlabel("Norm")
	plt.ylabel("Density")
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	if(out_path):
		plt.savefig(f'{out_path}.png')
	else:
	    plt.show()

def compute_density(weights,norm_fun):
    y_dict,max_value={},0
    for fun_i in norm_fun:
        norms_i=fun_i(weights).reshape(-1, 1)
        max_value=max(np.amax(norms_i),max_value)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kde.fit(norms_i)
        y_dict[str(fun_i)]=kde
    x=np.linspace(0,max_value, num=50).reshape(-1, 1)
    y_dict ={ id_i:np.exp(kde_i.score_samples(x))
                for id_i,kde_i in y_dict.items()}
    return x,y_dict

def mlp_paths(in_path):
	return {id_i:[path_i 
	         for id_i,path_i in utils.iter_files(path_i)
	             if(id_i=="MLP")][0]
	          for id_i,path_i in utils.iter_files(in_path)}

class WeightNorm(object):
	def __init__(self,dim=0):
		self.dim=dim

	def __str__(self):
	    return self.NAME 

class L1(WeightNorm):
    NAME="L1"
    def __call__(self,weights):
        weights=np.abs(weights)
        return np.sum(weights,axis=self.dim)

class L2(WeightNorm):
    NAME="L2"
    def __call__(self,weights):
        weights=weights**2
        norm=np.sum(weights,axis=self.dim)
        return np.sqrt(norm)

class LInf(WeightNorm):
	NAME="L_inf"
	def __call__(self,weights):
		return np.amax(weights,axis=self.dim)

class NormProp(object):
	def __init__(self, nom_norm,denom_norm,dim):
		self.nom_norm=nom_norm(dim)
		self.denom_norm=denom_norm(dim)
		self.name=f"{self.nom_norm}/{self.denom_norm}"

	def  __call__(self,weights):
		return self.nom_norm(weights)/self.denom_norm(weights)

	def __str__(self):
		return self.name

def norm_exp(in_path,out_path=None,dim=0):
	norm_fun=[L1,L2,LInf]
	norm_fun=[fun_i(dim) for fun_i in norm_fun]
	neural_dist(in_path,norm_fun,out_path)

def prop_exp(in_path,out_path=None,dim=0):
	norm_fun=[NormProp(LInf,L1,dim),
	          NormProp(L2,L1,dim),
	          NormProp(LInf,L2,dim)]
	neural_dist(in_path,norm_fun,out_path)

prop_exp(["AutoML/output","uci/output"],"norms/prop_rows",1)
