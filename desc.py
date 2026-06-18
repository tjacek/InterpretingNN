import dataset,utils

def make_desc(in_path):
    lines=[]
    paths=utils.top_files(in_path)
    for path_i in paths:
        data_i=dataset.read_csv(path_i)
        line_i=[path_i.split("/")[-1]]
        line_i.append(data_i.n_cats())
        line_i.append(len(data_i))
        line_i.append(data_i.dim())
        line_i.append(round(data_i.IR(),4))
        line_i+=list(data_i.pca_feats())
        print(line_i)
    print(paths)


make_desc("uci")