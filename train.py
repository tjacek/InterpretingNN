import base,dataset,deep


if __name__ == '__main__':
    data=dataset.read_csv("spatial/wine-quality-red")
    split=base.random_split(len(data),p=0.9)
    nn=deep.single_builder(params=data.params_dict())
#    nn.summary()
    result,_=split.eval(data,nn)
    print(result.get_acc())