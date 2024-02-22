import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import argparse

def plot_corr(df, data_name, savepath=None, show_plot=True): 
    corr = df.corr()
    cmap = plt.cm.seismic
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap=cmap)
    if savepath is not None:
        plt.savefig(savepath + '{}_correlation.png'.format(data_name))
    if show_plot:
        plt.show()

def plot_pairs(df, data_name, savepath=None, show_plot=True, sample=True):
    if show_plot or savepath is not None:
        df_ = df.sample(100) if sample else df
        sns.pairplot(df_, hue='y', plot_kws={'alpha': 0.5})
        if savepath is not None:
            plt.savefig(savepath + '{}_pairplot.png'.format(data_name))
        if show_plot:
            plt.show()

def get_savepath(loadpath):
    if not os.path.exists(loadpath + 'visuals/'):
        os.makedirs(loadpath + 'visuals/')
    return loadpath + 'visuals/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loadpath')
    parser.add_argument('name')
    args = parser.parse_args()

    loadpath = args.loadpath
    data_name = args.name
    df = pd.read_csv(loadpath + '{}.csv'.format(data_name))

    savepath = get_savepath(loadpath)

    plot_corr(df, data_name, savepath=savepath)
    plot_pairs(df, data_name, savepath=savepath)
