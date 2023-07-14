from utils.x_ai_tools import plot_clustering


reshape_l = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
reshape_label = lambda x: np.reshape(x, (x.shape[0] * x.shape[1]))

def plot_cluster_vis(data, labels, img_name, wandb_run, num_points):
    data = reshape_l(data)
    labels = reshape_label(labels)

    ##### Select a subset as these visualization tools are computationally expensive
    ind = np.random.permutation(l_vis_prior.shape[0])
    data = data[ind, :self.vis_dim]
    labels = labels[ind]
    data = data[:num_points]
    labels = labels[:num_points]

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print('Nans or Infs in the latent space of' + img_name)

    ####### Visualize the tsne/pca in matplotlib / pyplot

    #plot_clustering(l_vis_prior_concat, l_labels_concat, engine='matplotlib',
                    #exp_name=self._exp_name, img_name = self._run.name + '_' + str(i) + '_traintest_prior', wandb_run=self._run)
    plot_clustering(data,labels, engine='matplotlib',
                    exp_name=self._exp_name , img_name = img_name, wandb_run=wandb_run)

    

    