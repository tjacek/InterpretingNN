import matplotlib.pyplot as plt
import argparse
import train

def plot_xy( in_path,
	         x_clf="MLP",
	         y_clf="TabPFN"):
    output_dict=train.show_pred(in_path,verbose=False)
    x_dict=output_dict[x_clf]
    y_dict=output_dict[y_clf]
    fig=plt.figure()
    for data_i in x_dict:
        plt.text(x_dict[data_i], 
                 y_dict[data_i], 
                 data_i,
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel(x_clf)
    plt.ylabel(y_clf)
    plt.xlim(0.9*min(x_dict.values()),
             1.1*max(x_dict.values()))
    plt.ylim(0.9*min(y_dict.values()),
             1.1*max(y_dict.values()))
    plt.axline((0, 0), (1, 1))
    plt.grid()
    plt.show()

def show_plots(img_iter,out_path=None):
    for i,out_i in enumerate(img_iter):
#        error_hist(**out_i.__dict__)
        if(out_path):
            plt.savefig(f'{out_path}/{i}.png')
        else:
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=str,default="RF")
    parser.add_argument("--y", type=str,default="TabPFN")
    args=parser.parse_args()
    plot_xy(["AutoML/output","uci/output"],
    	    x_clf=args.x,y_clf=args.y)