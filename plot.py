import matplotlib.pyplot as plt
import seaborn as sn

def show_heatmap( matrix,
                  title,
                  out_path=None):
    sn.heatmap( matrix,
                cmap="YlGnBu",
                annot=False)#,
    plt.title(title)
    if(out_path):
        out_i=f"{out_path}/{title}"
        plt.tight_layout()
        plt.savefig(out_i,dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_txt(x_dict,
             y_dict,
             x_label,
             y_label):
    fig=plt.figure()
    for data_i in x_dict:
        plt.text(x_dict[data_i], 
                 y_dict[data_i], 
                 data_i,
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0.9*min(x_dict.values()),
             1.1*max(x_dict.values()))
    plt.ylim(0.9*min(y_dict.values()),
             1.1*max(y_dict.values()))
    plt.axline((0, 0), (1, 1))
    plt.grid()
    plt.show()