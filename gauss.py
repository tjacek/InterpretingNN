import numpy as np
from numpy import random
from dataclasses import dataclass

@dataclass
class Node(object):
	feat:int
	thres:float
	left:int
	right:int

    def __call__(self,x_i):
    	feat_i=x_i[self.feat]
    	if(feat_i< self.thres):
    		return self.left
    	else:
    		return self.right


class Tree(object):
	def __init__(self,nodes):
		self.nodes=nodes


def count_extr(x):
	count=np.sum(x>2)
	count+=np.sum(x< -2)
	return count

def gen_data(n,dims=100):
    return random.normal( loc=0, 
    	                  scale=1, 
    	                  size=(n, dims))