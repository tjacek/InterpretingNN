import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import argparse
import train,utils

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
        if(out_path):
            plt.savefig(f'{out_path}/{i}.png')
        else:
            plt.show()

def error_hist( names,
                y_true,
                y_pred,
                error):
    x = np.arange(len(names))
    plt.figure(figsize=(12, 6))

    plt.scatter(x, y_true,
                color="tab:blue",
                marker="o",
                label="True value")

    plt.errorbar(
        x,
        y_pred,
        yerr=error,
        fmt="s",
        color="tab:red",
        ecolor="black",
        elinewidth=1.5,
        capsize=5,
        label="Prediction ± std"
    )
    plt.xticks(x, names, rotation=90)
    plt.xlabel("Sample")
    plt.ylabel("Target")
    plt.title("Gaussian Process Regression - Leave-One-Out")
    plt.grid(alpha=0.7)
    plt.legend()
    plt.tight_layout()


def prediction_hist(names,
                    y_true,
                    y_pred):
    x = np.arange(len(names))
    plt.figure(figsize=(12, 6))

    plt.scatter(
        x,
        y_true,
        color="tab:blue",
        marker="o",
        label="True value"
    )

    plt.scatter(
        x,
        y_pred,
        color="tab:red",
        marker="s",
        label="Prediction"
    )

    plt.xticks(x, names, rotation=90)
    plt.xlabel("Sample")
    plt.ylabel("Target")
#    plt.title("Regression - Leave-One-Out")
    plt.grid(alpha=0.7)
    plt.legend()
    plt.tight_layout()

def show_heatmap(in_path,out_path=None):
    if(out_path):
        utils.make_dir(out_path)
    for id_i,path_i in utils.iter_files(in_path):
        values = np.loadtxt(path_i)
#        values=utils,norm_matrix(values)
#        values/=np.sum(values,axis=0)
#        sn.heatmap(values)
        sn.heatmap(values,
                   cmap="YlGnBu",
                   annot=False)#,
        plt.title(id_i)
        if(out_path):
            out_i=f"{out_path}/{id_i}"
            plt.tight_layout()
            plt.savefig(out_i,dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def get_matrix_dict(in_path):
    return { id_i:np.loadtxt(path_i) 
                for id_i,path_i in utils.iter_files(in_path)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,default="matrix/infl")
    parser.add_argument("--output", type=str,default="matrix/inf_plot")
    args=parser.parse_args()
    show_heatmap(args.input,
                 args.output)
#    parser.add_argument("--x", type=str,default="RF")
#    parser.add_argument("--y", type=str,default="TabPFN")
#    args=parser.parse_args()
#    plot_xy(["AutoML/output","uci/output"],
#    	    x_clf=args.x,y_clf=args.y)