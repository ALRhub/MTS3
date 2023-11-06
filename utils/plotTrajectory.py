import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
import tueplots
from tueplots import bundles
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils.dataProcess import denorm,denorm_var
from hydra.utils import get_original_cwd, to_absolute_path
import wandb
from matplotlib import rc
#rc("text", usetex=False)
matplotlib.rcParams['axes.unicode_minus'] = False #https://github.com/garrettj403/SciencePlots/issues/2

def plotJoints(gt,pred_mu,pred_std, valid_flag, traj, wandb_run, show=False, exp_name='trial'):
    if gt.shape[1] < 2:
        fig, axs = plt.subplots(1,2)
        return fig, axs

    n1 = int((gt.shape[-1])/2)+1
    n2 = 2
    #with plt.style.context(['science']: #https://github.com/garrettj403/SciencePlots/blob/master/examples/plot-examples.py
    #with plt.rc_context(bundles.jmlr2001()):
    fig, axs = plt.subplots(n1, n2)
    fig.set_size_inches(8, 6)  ##set the size of the figure important for nicer plot saving
    dim = 0
    for i in range(n1):
        for j in range(n2):
            if dim >= gt.shape[-1]:
                break
            gt_dim = gt[:,dim]
            pred_mu_dim = pred_mu[:, dim]
            if pred_std is not None:
                pred_std_dim = pred_std[:, dim]
            dim = dim + 1

            axs[i, j].plot(gt_dim)
            if valid_flag is not None:
                axs[i, j].scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],
                            gt_dim[np.logical_not(valid_flag)], facecolor='red', s=2)
            axs[i, j].plot(pred_mu_dim, color='black')
            axs[i, j].set_title('Joint '+ str(dim),y = 1.0, pad = -14)
            if pred_std is not None:
                    axs[i, j].fill_between(np.arange(len(gt)), pred_mu_dim - pred_std_dim, pred_mu_dim + pred_std_dim, alpha=0.2, color='grey')

            folder_name = get_original_cwd() + '/logs/latent_plots'
    if show == True:
        plt.show()
        plt.close()
    else:
        #split exp_name by '/' and take the last one
        exp_name_split = exp_name.split('/')
        print(exp_name_split)
        #plt.show(block=False)

        plt.savefig(folder_name + "/traj_" + str(traj) + '_' + exp_name_split[-1] + ".png", bbox_inches='tight')
        image = plt.imread(folder_name + "/traj_" + str(traj) + '_' + exp_name_split[-1] + ".png")
        if wandb_run is not None:
            key = 'Trajectory_' + str(traj) + '_Step_' + exp_name_split[-2] + "_alog_" + exp_name_split[-1] + "_type_" + exp_name_split[0]
            wandb_run.log({ key: wandb.Image(image)})
            os.remove(folder_name + "/traj_" + str(traj) + '_' +  exp_name_split[-1] + ".png")
            plt.close()

    return fig,axs







def plotImputation(gts, valid_flags, pred_mus, pred_vars, wandb_run, l_priors=None, l_posts=None, task_labels=None, num_traj: int =2, log_name='test', exp_name='trial', show=False, latent_Vis=False):
    # make a folder to save the plots
    if type(gts) is not np.ndarray:
        gts = gts.cpu().detach().numpy()
    if type(valid_flags) is not np.ndarray:
        valid_flags = valid_flags.cpu().detach().numpy()
    if type(pred_mus) is not np.ndarray:
        pred_mus = pred_mus.cpu().detach().numpy()
    if type(pred_vars) is not np.ndarray:
        pred_vars = pred_vars.cpu().detach().numpy()
    if not os.path.exists(get_original_cwd() + "/logs/output/plots/" + exp_name):
        os.makedirs(get_original_cwd() + "/logs/output/plots/" + exp_name)
    np.savez(get_original_cwd() + "/logs/output/plots/" + exp_name + "/gt", gts[:100])
    np.savez(get_original_cwd() + "/logs/output/plots/" + exp_name + "/pred_mean_" + str(wandb_run.id), pred_mus[:100])
    np.savez(get_original_cwd() + "/logs/output/plots/" + exp_name + "/valid_flags_" + str(wandb_run.id), valid_flags[:100])
    np.savez(get_original_cwd() + "/logs/output/plots/" + exp_name + "/pred_var_" + str(wandb_run.id), pred_vars[:100])

    trjs = np.random.randint(gts.shape[0],size=num_traj)
    n=0
    for traj in trjs:
        gt = gts[traj,:,:]
        pred_mu = pred_mus[traj, :, :]
        if valid_flags is not None:
            valid_flag = valid_flags[traj, :, 0]
        else:
            valid_flag = None
        if pred_vars is not None:
            pred_var = pred_vars[traj, :, :]
        else:
            pred_var = None
        fig, axs = plotJoints(gt,pred_mu,pred_var,valid_flag,wandb_run=wandb_run,traj=traj,show=show, exp_name=exp_name)

        folder_name = get_original_cwd() + '/logs/latent_plots'
        


def plotImputationDiff(gts, valid_flags, pred_mus, pred_stds, wandb_run, dims=[0,1,2,3,4,5], num_traj: int =2, log_name='test', exp_name='trial', show=False):
    folder_name = get_original_cwd() + '/logs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            if valid_flags is not None:
                valid_flag = valid_flags[traj,:,0]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            if valid_flags is not None:
                plt.scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],gt[np.logical_not(valid_flag)],facecolor='red',s=14)
            plt.plot(pred_mu, color='black')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'Traj' + str(traj) + '_dim_' + str(dim) +'_' + exp_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()

def plotLongTerm(gts, pred_mus, pred_stds, wandb_run, dims=[0], num_traj=2, log_name='test', exp_name='trial', show=False):
    folder_name = get_original_cwd() + '/logs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            plt.plot(pred_mu, color='black')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'MultiStep_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()


def plotMbrl(gts, pred_mus, pred_stds, wandb_run, dims=[0,1,2,3], num_traj=2, log_name='test', exp_name='trial', show=False):
    folder_name = os.getcwd() + '/logs/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            plt.plot(pred_mu, color='black')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'MBRL_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()



if __name__ == '__main__':
    global ax
    gt = np.random.rand(10,50,1)
    pred = np.random.rand(10,50,1)
    std = np.random.uniform(low=0.01, high=0.1, size=(10,50,1))
    rs = np.random.RandomState(seed=23541)
    obs_valid = rs.rand(gt.shape[0], gt.shape[1], 1) < 1 - 0.5
    pred = np.random.rand(10, 50, 1)
    plotSimple(gt[1,:,0],obs_valid[1,:,0],pred[1,:,0],pred_std=std[1,:,0])
    plotMbrl(gt[1,:,0],pred[1,:,0],pred_std=std[1,:,0])