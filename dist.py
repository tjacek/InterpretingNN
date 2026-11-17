import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

class EmpDist:
	def __init__( self,
		          values,
		          density):
		self.values=values
		self.density=density

	@classmethod
	def make(cls,points,range=5):
		points=points.reshape(-1,1)
		kde=KernelDensity(kernel='gaussian', bandwidth=0.2)
		kde.fit(points)
		x=np.linspace(-range,range, num=50).reshape(-1, 1)
		y=np.exp(kde.score_samples(x))
		return cls(x.flatten(),y)
    
	def __call__(self,type=None):
		if(type=="pos"):
			return self.pos_log()
		elif(type=="neg"):
			return self.neg_log()
		elif(type=="fourrier"):
			return self.fourrier()
		else:
			return self.values,self.density

	def plot( self,
		      title="",
		      out_path=None,
		      type=None):
		x,y=self(type)
		plt.plot(x,y)
		plt.title(title)
		plt.xlabel("Values")
		plt.ylabel("Density")
		return plt

	def kurt(self):
		return kurtosis(self.density)

	def pos_log(self):
		pos=(self.values>0)
		x=np.log(self.values[pos])
		y=np.log(self.density[pos])
		return x,y

	def neg_log(self):
		neg=(self.values<0)
		x=np.log(self.values[neg])
		y=np.log(self.density[neg])	
		return x,y

	def fourrier(self):
		f = np.asarray(self.density, dtype=float)
		N = len(f)
		F = np.fft.rfft(f)
		freqs = np.fft.rfftfreq(N, d=1)
		amp = np.abs(F) / N
		if N % 2 == 0:
#			x=self.values[1:-1]
			amp[1:-1] *= 2
		else:
#			x=self.values[1:]
			amp[1:] *= 2
		return freqs,amp