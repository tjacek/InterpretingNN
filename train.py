import base,clf,dataset,deep,utils

def basic_exp(in_path):
    clf_types= clf.Clf.to_dict()
    print(clf_types)
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        splits_i=base.SplitGroup.make( data_i,
                                       n_repeats=1,
                                       n_splits=10)
        for name_j,clf_type_j in clf_types.items():
            print(name_j)
            results=splits_i(data_i,clf_type_j)
            print(results.get_acc())

def make_models(in_path,out_path):
    clf_types=[clf.RF,clf.GRAD,clf.LR,clf.SVM]
    utils.make_dir(out_path)
    for id_i,path_i in utils.iter_files(in_path):
        print(path_i)
        data_i=dataset.read_csv(path_i)
        splits_i=base.SplitGroup.make( data_i,
                                       n_repeats=1,
                                       n_splits=10)
        out_i=f"{out_path}/{id_i}"
        utils.make_dir(out_i)
        splits_i.save(f"{out_i}/splits")
        for type_j in clf_types:
            out_ij=f"{out_i}/{type_j.NAME}"
            results=splits_i(data_i,type_j)
            results.save(out_ij)
            print(results.get_acc())

if __name__ == '__main__':
    make_models("uci","output")