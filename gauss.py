import numpy as np
from numpy import random
from dataclasses import dataclass
import itertools

@dataclass
class Node(object):
    feat:int
    thres:float
    left:"Node"
    right:"Node"
    
    def is_leaf(self):
        return False	

    def __call__(self,x_i):
        feat_i=x_i[self.feat]
        if(feat_i< self.thres):
            return self.left
        else:
            return self.right

    @classmethod
    def random(cls,dims=100):
        feats=random.randint(dims)
        thres=float(random.randint(2)-1)
        return cls(feats,thres,None,None) 

@dataclass
class Leaf(object):
    cat:int

    def is_leaf(self):
        return True	

    @classmethod
    def random(cls,n_cats=10):
        cats=random.randint(n_cats)
        return cls(cats)

class Tree(object):
    def __init__(self,
                 root,
                 nodes):
        self.root=root
        self.nodes=nodes
    
    def __call__(self,x):
        node=self.root
        while(not node.is_leaf()):
            node=node(x)
        return node.cat

    @classmethod
    def random( cls,
                levels=3,
                n_cats=10,
                dims=100):
        root=Node.random(dims)
        all_nodes=[[root]]
        for i in range(levels):
            old_nodes=all_nodes[-1]
            new_nodes=[]
            for node_j in old_nodes:
                node_j.right=Node.random(dims)
                node_j.left=Node.random(dims)
                new_nodes.append(node_j.right)
                new_nodes.append(node_j.left)
            all_nodes.append(new_nodes)
        for node_i in all_nodes[-1]:
            node_i.right=Leaf.random(n_cats)
            node_i.left=Leaf.random(n_cats)
        nodes=list(itertools.chain(*all_nodes))
        return cls(root,nodes)

def count_extr(x):
	count=np.sum(x>2)
	count+=np.sum(x< -2)
	return count

def gen_data(n,dims=100):
    return random.normal( loc=0, 
    	                  scale=1, 
    	                  size=(n, dims))


tree=Tree.random()
print(tree)