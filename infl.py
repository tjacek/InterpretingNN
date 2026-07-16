from tqdm import tqdm
import base,clf,dataset,utils

@utils.dir_fun
def infl_rf(in_path,out_path=None):
    print(in_path)
    data_i=dataset.read_csv(in_path)
    split_i= base.Split.random(data_i)
    acc=eval_split(split_i,data_i)
    infl_i=[]
    for j in tqdm(range(data_i.dim())):
    	data_j=data_i.remove_col(j)
    	acc_j=eval_split(split_i,data_i)
    	infl_i.append(acc-acc_j)
    print(infl_i)

def eval_split(split_i,data_i):
    clf_i=clf.RF()
    split_i.fit_clf(data_i,clf_i)
    result_i=split_i.pred(data_i,clf_i)
    return result_i.get_acc()

infl_rf("uci/data","test")