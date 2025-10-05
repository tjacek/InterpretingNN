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
    
    def var_contr(self,model,n=1000):
        x=self(n)
        y=model.predict_proba(x)
        all_sqr=[]
        for i,x_i in enumerate(x):
            delta_x_i=self.change(x,0,n)
            delta_y_i=model.predict_proba(delta_x_i)
            y_i=y[i]
            for y_j in delta_y_i:
                err=y_i-y_j
#                print(err)
                sqr_err=err**2
                all_sqr.append(sqr_err)
        print(np.mean(all_sqr,axis=0))

    def change(self,x,cord_i,n):
        max_i,min_i=self.bounds[cord_i]
        new_x=[]
        for _ in range(n):
            x_j=x.copy()
            x_j[cord_i]=np.random.uniform(max_i,min_i)
            new_x.append(x_j)
        return np.array(x_j)

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

def random_exp(in_path,p):
    data=dataset.read_csv(in_path)
    split=base.random_split(len(data),p=p)
    sampler=RandomSample(data.range())
    nn=deep.single_builder(params=data.params_dict())
    split=base.random_split(len(data),p=0.9)
    result,_=split.eval(data,nn)
    sampler.var_contr(nn)
#    E,Var=sampler.iter_sampling(nn)
#    print(E)
#    print(Var)

#    y_prob=nn.predict_proba(X)
#    print(np.mean(y_prob,axis=0))
#    print(np.std(y_prob,axis=0))

if __name__ == '__main__':
    random_exp("wine-quality-red",p=0.9)