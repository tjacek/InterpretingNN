import base,dataset,deep


if __name__ == '__main__':
    data=dataset.read_csv("spatial/wine-quality-red")
    split=base.random_split(len(data),p=0.9)
#    nn=deep.single_builder(params=data.params_dict())
#    nn.summary()
    nn=deep.make_mlp(data)
    result,_=split.eval(data,nn)
    print(f"Unbalanced:{result.get_acc():.4f}")
    nn=deep.make_balanced_mlp(data)
    result,_=split.eval(data,nn)
    print(f"Balanced:{result.get_acc():.4f}")