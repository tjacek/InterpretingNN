import argparse
#import clf
import exp
import utils

def ablat_exp(in_path):
    conf=utils.read_json(in_path)
    proto=exp.ExpParams( conf["data_path"],      
                         conf["split_path"],
                         conf["out_path"])
    clf_type=proto.get_clf()
    utils.make_dir(proto.out_path)
    data,splits=proto.get_data()
    for i in range(data.dim()):
        data_i=data.remove_col(i)
        result_i,_=splits( data,
                           clf_type,
                           verbose=True)
        out_i=f"{proto.out_path}/{i}"
        result_i.save(out_i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path",type=str,default="selected/conf.json") 
    args=parser.parse_args()
    ablat_exp(args.conf_path)