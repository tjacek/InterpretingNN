import base,clf,dataset,deep,utils

def basic_exp(in_path):
    clf_types= clf.Clf.to_dict()
    print(clf_types)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        splits_i=base.SplitGroup.get_split( data_i,
                                          n_repeats=1,
                                          n_splits=10)
#        split_i=dataset.Split.random(data_i,p=0.9)
        for name_j,clf_type_j in clf_types.items():
            print(name_j)
            results=splits_i(data_i,clf_type_j)
            print(results.get_acc())
#            clf_j=clf_type_j()
#            split_i.fit_clf(data_i,clf_j)
#            result_j=split_i.pred(data_i,clf_j)

#            print(result_j.get_acc())

if __name__ == '__main__':
    basic_exp("uci")
#    data=dataset.read_csv("spatial/wine-quality-red")
#    split=base.random_split(len(data)wine,p=0.9)
#    nn=deep.make_mlp(data)
#    result,_=split.eval(data,nn)
#    print(f"Unbalanced:{result.get_acc():.4f}")
#    nn=deep.make_balanced_mlp(data)
#    result,_=split.eval(data,nn)
#    print(f"Balanced:{result.get_acc():.4f}")