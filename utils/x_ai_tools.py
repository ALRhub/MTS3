import os
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pandas as pd
import umap

from plotly.graph_objs import *
import plotly
from hydra.utils import get_original_cwd, to_absolute_path



def plot_clustering(z_run, labels, components=2, engine ='plotly', download = False, folder_name=None, exp_name =None, img_name = 'trial', wandb_run=None):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    def plot_clustering_plotly(z_run, labels):
        reducer = umap.UMAP(n_components=components)

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF)) 

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=components+1).fit_transform(z_run)
        z_run_tsne = TSNE(n_components=components, perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        z_run_umap = reducer.fit_transform(z_run)

        if components==2:
            trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
            )
        else:
            trace = Scatter(
                x=z_run_pca[:, 0],
                y=np.array(np.arange(z_run_pca.shape[0])),
                mode='markers',
                marker=dict(color=colors)
            )
        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        if components==2:
            trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
            )
        else:
            trace = Scatter(
                x=z_run_tsne[:, 0],
                y=np.array(np.arange(z_run_tsne.shape[0])),
                mode='markers',
                marker=dict(color=colors)
            )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        if components==2:
            trace = Scatter(
            x=z_run_umap[:, 0],
            y=z_run_umap[:, 1],
            mode='markers',
            marker=dict(color=colors)
            )
        else:
            trace = Scatter(
                x=z_run_umap[:, 0],
                y=np.array(np.arange(z_run_umap.shape[0])),
                mode='markers',
                marker=dict(color=colors)
            )

        data = Data([trace])
        layout = Layout(
            title='UMAP on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download=False, folder_name=None, exp_name='trial', img_name="img"):
        print(">>>>>>>>>",components)
        labels = labels[:z_run.shape[0]] # because of weird batch_size
        #labels = labels.astype(str)
        if folder_name is None:
            folder_name = get_original_cwd() + '/plots/latent_plots'
        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        reducer = umap.UMAP(n_components=components)

        #colors = [hex_colors[int(i)] for i in labels]
        print(components)
        if z_run.shape[-1]<=2:
            z_run_pca = z_run
            z_run_tsne = z_run
            z_run_umap = z_run
        else:
            #z_run_pca = TruncatedSVD(n_components=components).fit_transform(z_run)
            z_run_tsne = TSNE(n_components=components, perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
            z_run_umap = reducer.fit_transform(z_run)
            z_run_pca = z_run_tsne

        if components==2:
            df = pd.DataFrame({'dim1': z_run_pca[:, 0],'dim2':z_run_pca[:, 1],  'labels': labels})
        else:
            print(">>>>>>>>>", labels.shape,z_run_pca.shape)
            df = pd.DataFrame({'dim1': np.array(np.arange(z_run_tsne.shape[0])), 'dim2': z_run_pca[:, 0], 'labels': labels})

        ax = sns.scatterplot(x='dim1', y='dim2', hue=labels, palette="RdBu", data=df)
        norm = plt.Normalize(df['labels'].min(), df['labels'].max())
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        ax.figure.colorbar(sm)

        #plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
        #plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca_" + exp_name +".png")
        else:
            #plt.show()
            plt.savefig(folder_name + "/pca_" + exp_name + ".png")
            image = plt.imread(folder_name + "/pca_" + exp_name + ".png")
            if wandb_run is not None:
                key= 'PCA_' + img_name + "_" + str(wandb_run.step)
                wandb_run.log({key: wandb.Image(image)})
                os.remove(folder_name + "/pca_" + exp_name + ".png")

        fig, ax = plt.subplots()

        if components==2:
            df = pd.DataFrame({'dim1': z_run_tsne[:, 0], 'dim2': z_run_tsne[:, 1], 'labels': labels})
        else:
            print(z_run_tsne[:, 0])
            df = pd.DataFrame({'dim1': np.array(np.arange(z_run_tsne.shape[0])), 'dim2': z_run_tsne[:, 0], 'labels': labels})


        ax = sns.scatterplot(x='dim1', y='dim2', hue=labels, palette="RdBu", data=df)
        norm = plt.Normalize(df['labels'].min(), df['labels'].max())
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        ax.figure.colorbar(sm)
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne_" + exp_name +".png")
        else:
            plt.savefig(folder_name + "/tsne_" + exp_name + ".png")
            image = plt.imread(folder_name + "/tsne_" + exp_name + ".png")
            #plt.show()
            if wandb_run is not None:
                key = 'TSNE_' + img_name + "_" + str(wandb_run.step)
                wandb_run.log({key: wandb.Image(image)})
                os.remove(folder_name + "/tsne_" + exp_name + ".png")

        fig, ax = plt.subplots()
        if components==2:
            df = pd.DataFrame({'dim1': z_run_umap[:, 0], 'dim2': z_run_umap[:, 1], 'labels': labels})
        else:
            df = pd.DataFrame({'dim1': np.array(np.arange(z_run_umap.shape[0])), 'dim2': z_run_umap[:, 0], 'labels': labels})

        ax = sns.scatterplot(x='dim1', y='dim2', hue=labels, palette="RdBu", data=df)
        norm = plt.Normalize(df['labels'].min(), df['labels'].max())
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        sm.set_array([])

        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        ax.figure.colorbar(sm)
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/umap_" + exp_name + ".png")
        else:
            plt.savefig(folder_name + "/umap_" + exp_name + ".png")
            image = plt.imread(folder_name + "/umap_" + exp_name + ".png")
            # plt.show()
            if wandb_run is not None:
                key = 'UMAP_' + img_name + "_" + str(wandb_run.step)
                wandb_run.log({key: wandb.Image(image)})
                os.remove(folder_name + "/umap_" + exp_name + ".png")


    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name, exp_name, img_name)

def plot_clustering_1d(z_run, labels, components=2, engine ='plotly', download = False, folder_name =None, exp_name = 'trial', wandb_run=None):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    def plot_clustering_plotly(z_run, labels):
        reducer = umap.UMAP(n_components=components)

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=components+1).fit_transform(z_run)
        z_run_tsne = TSNE(n_components=components, perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        z_run_umap = reducer.fit_transform(z_run)

        if components==2:
            trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
            )
        else:
            trace = Scatter(
                x=z_run_pca[:, 0],
                y=np.array(np.arange(z_run_pca.shape[0])),
                mode='markers',
                marker=dict(color=colors)
            )
        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        if components==2:
            trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
            )
        else:
            trace = Scatter(
                x=z_run_tsne[:, 0],
                y=np.array(np.arange(z_run_tsne.shape[0])),
                mode='markers',
                marker=dict(color=colors)
            )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        if components==2:
            trace = Scatter(
            x=z_run_umap[:, 0],
            y=z_run_umap[:, 1],
            mode='markers',
            marker=dict(color=colors)
            )
        else:
            trace = Scatter(
                x=z_run_umap[:, 0],
                y=np.array(np.arange(z_run_umap.shape[0])),
                mode='markers',
                marker=dict(color=colors)
            )

        data = Data([trace])
        layout = Layout(
            title='UMAP on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download=False, folder_name=None, exp_name='trial'):
        print(">>>>>>>>>",components)
        labels = labels[:z_run.shape[0]] # because of weird batch_size
        #labels = labels.astype(str)
        if folder_name is None:
            folder_name = get_original_cwd() + '/plots/latent_plots'
        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        reducer = umap.UMAP(n_components=components)

        #colors = [hex_colors[int(i)] for i in labels]
        #print(components)
        if z_run.shape[-1]<=1:
            z_run_pca = z_run
            z_run_tsne = z_run
            z_run_umap = z_run
        else:
            z_run_pca = TruncatedSVD(n_components=components).fit_transform(z_run)
            z_run_tsne = TSNE(n_components=components, perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
            z_run_umap = reducer.fit_transform(z_run)
        #print(z_run_pca)
        #print(">>>>>>>>>>>>>>>>",z_run.shape)

        plt.figure(figsize=(12.5,7))
        times =z_run_tsne.shape[0]
        # times = z_run_pca[:20, 0].shape[0]
        print(">>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>",times)
        df = pd.DataFrame({'time (0.30s)': np.array(np.arange(times)), 'inferred_task_pc1': z_run_pca[:times, 0], 'inferred_task_pc2': z_run_pca[:times, 0], 'GT_task': labels[:times]})
        sns.set(rc={'figure.figsize': (12, 3)})
        plt.subplot(2, 1, 1)
        ax = sns.lineplot(x='time (0.30s)', y='inferred_task_pc1', linewidth=3, hue=None, legend="brief", data=df)
        ax.set_xlabel('', fontsize=25)
        ax.set_ylabel('Inferred Task', fontsize=25)
        # ax = sns.lineplot(x='time (0.30s)', y='inferred_task_pc2', linewidth=3, hue=None, legend="brief", data=df)
        # ax.set_xlabel('time (0.30s)', fontsize=25)
        plt.subplot(2, 1, 2)
        ax = sns.lineplot(x='time (0.30s)', y='GT_task', hue=None, linewidth=3, color="red", legend="brief", data=df)
        ax.set_xlabel('time (0.30s)', fontsize=25)
        ax.set_ylabel('GT Task', fontsize=25)
        # plt.gcf().text(0.5,-0.04,"time (0.15 ms)", ha="center")


        # ax = sns.scatterplot(x='dim1', y='dim2', hue=None, palette="RdBu", data=df)
        # norm = plt.Normalize(df['labels'].min(), df['labels'].max())
        # sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        # sm.set_array([])

        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        # ax.figure.colorbar(sm)

        #plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
        #plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca_" + exp_name +".png")
        else:
            #plt.show()
            plt.savefig(folder_name + "/pca_" + exp_name + ".png")
            image = plt.imread(folder_name + "/pca_" + exp_name + ".png")
            if wandb_run is not None:
                key= 'PCA_' + str(wandb_run.step)
                wandb_run.log({key: wandb.Image(image)})
                os.remove(folder_name + "/pca_" + exp_name + ".png")

        fig, ax = plt.subplots()

        plt.figure(figsize=(12.5, 7))

        df = pd.DataFrame(
            {'time (0.30s)': np.array(np.arange(times)), 'inferred_task_pc1': z_run_tsne[:times, 0],
                'inferred_task_pc2': z_run_tsne[:times, 0], 'GT_task': labels[:times]})
        sns.set(rc={'figure.figsize': (12, 3)})
        plt.subplot(2, 1, 1)
        ax = sns.lineplot(x='time (0.30s)', y='inferred_task_pc1', linewidth=3, hue=None, legend="brief", data=df)
        ax.set_xlabel('', fontsize=25)
        ax.set_ylabel('Inferred Task', fontsize=25)
        # ax = sns.lineplot(x='time (0.30s)', y='inferred_task_pc1', linewidth=3, hue=None, legend="brief", data=df)
        # ax.set_xlabel('time (0.30s)', fontsize=25)
        # ax.set_ylabel('Inferred Task', fontsize=25)
        plt.subplot(2, 1, 2)
        ax = sns.lineplot(x='time (0.30s)', y='GT_task', hue=None, linewidth=3, color="red", legend="brief", data=df)
        ax.set_xlabel('time (0.30s)', fontsize=25)
        ax.set_ylabel('GT Task', fontsize=25)

        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        # ax.figure.colorbar(sm)
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne_" + exp_name +".png")
        else:
            plt.savefig(folder_name + "/tsne_" + exp_name + ".png")
            image = plt.imread(folder_name + "/tsne_" + exp_name + ".png")
            #plt.show()
            if wandb_run is not None:
                key = 'TSNE_' + str(wandb_run.step)
                wandb_run.log({key: wandb.Image(image)})
                os.remove(folder_name + "/tsne_" + exp_name + ".png")

        plt.figure(figsize=(12.5, 7))
        df = pd.DataFrame(
            {'time (0.30s)': np.array(np.arange(times)), 'inferred_task_pc1': z_run_umap[:times, 0],
                'inferred_task_pc2': z_run_umap[:times, 0], 'GT_task': labels[:times]})
        sns.set(rc={'figure.figsize': (12, 3)})
        plt.subplot(2, 1, 1)
        ax = sns.lineplot(x='time (0.30s)', y='inferred_task_pc1', linewidth=3, hue=None, legend="brief", data=df)
        ax.set_xlabel('', fontsize=25)
        ax.set_ylabel('Inferred Task', fontsize=25)
        # ax = sns.lineplot(x='time (0.30s)', y='inferred_task_pc1', linewidth=3, hue=None, legend="brief", data=df)
        # ax.set_xlabel('time (0.30s)', fontsize=25)
        plt.subplot(2, 1, 2)
        ax = sns.lineplot(x='time (0.30s)', y='GT_task', hue=None, linewidth=3, color="red", legend="brief", data=df)
        ax.set_xlabel('time (0.30s)', fontsize=25)
        ax.set_ylabel('GT Task', fontsize=25)


        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        # ax.figure.colorbar(sm)
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/umap_" + exp_name + ".png")
        else:
            plt.savefig(folder_name + "/umap_" + exp_name + ".png")
            image = plt.imread(folder_name + "/umap_" + exp_name + ".png")
            # plt.show()
            if wandb_run is not None:
                key = 'UMAP_' + str(wandb_run.step)
                wandb_run.log({key: wandb.Image(image)})
                os.remove(folder_name + "/umap_" + exp_name + ".png")


    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name, exp_name)

def vis_1d(z_run, components=2):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    #labels = labels.astype(str)

    reducer = umap.UMAP(n_components=components)

    #colors = [hex_colors[int(i)] for i in labels]
    #print(components)
    if z_run.shape[-1]<=1:
        z_run_pca = z_run
        z_run_tsne = z_run
        z_run_umap = z_run
    else:
        z_run_pca = TruncatedSVD(n_components=components).fit_transform(z_run)
        z_run_tsne = TSNE(n_components=components, perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)
        z_run_umap = reducer.fit_transform(z_run)

    return z_run_pca, z_run_tsne, z_run_umap



def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]