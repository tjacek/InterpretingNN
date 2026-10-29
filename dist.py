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
		return cls(x,y)

	def plot(self,title=""):
		plt.plot(self.values, self.density)
		plt.title(title)
		plt.xlabel("Values")
		plt.ylabel("Density")
		plt.show()

	def kurt(self):
		return kurtosis(self.density)