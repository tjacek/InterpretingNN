import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import deep, utils

def neural_dist(in_path):
	paths=mlp_paths(in_path)
	for id_i,path_i in paths.items():
		mlp_i=deep.MLP.read(f"{path_i}/models/0.keras")
		weights=mlp_i.get_weights()
		x,y_dict=compute_density(weights,[L1(),L2(),LInf()])
		plot(x,y_dict,id_i)

def plot(x,y_dict,title):
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

class L1(object):
    def __call__(self,weights):
        weights=np.abs(weights)
        return np.sum(weights,axis=0)

    def __str__(self):
    	return "L1"

class L2(object):
    def __call__(self,weights):
        weights=weights**2
        norm=np.sum(weights,axis=0)
        return np.sqrt(norm)

    def __str__(self):
    	return "L2"

class LInf(object):
	def __call__(self,weights):
		return np.amax(weights,axis=0)
    
	def __str__(self):
		return "L_inf"

neural_dist("uci/output")

#weights=np.array([[4,3],
#	              [3,4]])

#print(L1()(weights))
#print(L2()(weights))
