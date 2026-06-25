import clf,dataset,deep,utils

def basic_exp(in_path):
    clf_types= clf.Clf.to_dict()
    print(clf_types)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        split_i=dataset.Split.random(data_i,p=0.9)
        for name_j,clf_type_j in clf_types.items(): 
            clf_j=clf_type_j()
            print(clf_j)

if __name__ == '__main__':
    basic_exp("uci")
#    data=dataset.read_csv("spatial/wine-quality-red")
#    split=base.random_split(len(data),p=0.9)
#    nn=deep.make_mlp(data)
#    result,_=split.eval(data,nn)
#    print(f"Unbalanced:{result.get_acc():.4f}")
#    nn=deep.make_balanced_mlp(data)
#    result,_=split.eval(data,nn)
#    print(f"Balanced:{result.get_acc():.4f}")