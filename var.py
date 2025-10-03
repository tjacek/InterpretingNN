import numpy as np 
import base,dataset,deep

class RandomSample(object):
    def __init__(self,bounds):
        self.bounds=bounds

    def __call__(self,n=1000):
        X=[]
        for max_i,min_i in self.bounds:
            x_i=np.random.uniform(max_i, min_i, n)
            X.append(x_i)
        return np.array(X).T

def random_exp(in_path,p):
    data=dataset.read_csv(in_path)
    split=base.random_split(len(data),p=p)
    sampler=RandomSample(data.range())
    X=sampler()
    print(X.shape)
    nn=deep.single_builder(params=data.params_dict())
    split=base.random_split(len(data),p=0.9)
    result,_=split.eval(data,nn)
    y_prob=nn.predict_proba(X)
    print(np.mean(y_prob,axis=0))
    print(np.std(y_prob,axis=0))

if __name__ == '__main__':
    random_exp("wine-quality-red",p=0.9)