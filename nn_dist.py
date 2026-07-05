import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import deep, utils

def neural_dist(in_path):
	paths=mlp_paths(in_path)
	for id_i,path_i in paths.items():
		mlp_i=deep.MLP.read(f"{path_i}/models/0.keras")
		weights=mlp_i.get_weights()
		norms=L1(weights).reshape(-1, 1)
		kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(norms)
		x=np.linspace(0,np.amax(norms), num=50).reshape(-1, 1)
		y = np.exp(kde.score_samples(x))

		plt.figure(figsize=(8, 5))
		plt.plot(
            x[:, 0],
            y,
            label=str(id_i)
        )
		plt.xlabel("L1 norm")
		plt.ylabel("Density")
		plt.title("Kernel density of neural weights")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()

def mlp_paths(in_path):
	return {id_i:[path_i 
	         for id_i,path_i in utils.iter_files(path_i)
	             if(id_i=="MLP")][0]
	          for id_i,path_i in utils.iter_files(in_path)}

def L1(weights):
    weights=np.abs(weights)
    return np.sum(weights,axis=0)

neural_dist("uci/output")