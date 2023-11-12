import sys

sys.path.append('.')
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.metrics import root_mean_squared, joint_rmse, gaussian_nll
from tueplots import bundles
import pickle
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils.latentVis import vis_1d
from hydra.utils import get_original_cwd, to_absolute_path
import wandb
from utils.dataProcess import norm, denorm, denorm_var


def plotAlgo(ax, algo, exp_name, step_name, episode_length, traj_num, joint_num, var_plot=False, show=norm):
    print(".........................................", algo)
    folder_name = os.getcwd() + "/experiments/output/plots/" + exp_name + '/' + str(step_name)

    ## loop over all the subfolders in a folder path
    firstPlot = norm
    subdir = algo
    subpath = os.path.join(folder_name, subdir)
    id_list = []
    for file in os.listdir(subpath):
        if len(id_list) >= 1:
            break
        ## loop over files in subpath
        if file == "gt.npz" or file == "normalizer.npz" or file == "valid_flags_1.npz":
            continue
        else:
            id = file.split("_")[-1].split(".")[0]
            if id in id_list:
                continue
            id_list.append(id)
        if firstPlot:
            if subdir != "Transformer" and subdir != "Transformer-2" and subdir != "AR-Transformer" and subdir != "Multi-Transformer" and subdir != "AR" and subdir != "Multi":
                flag_name = "valid_flags_" + str(id) + ".npz"
                valid_flags = np.load(os.path.join(subpath, flag_name))['arr_0']
                gt_name = "gt.npz"
                gts = np.load(os.path.join(subpath, gt_name))['arr_0']
                gt = gts[traj_num,:,joint_num]
                valid_flag = valid_flags[traj_num,:]
            else:
                flag_name = "valid_flags_" + str(1) + ".npz"
                valid_flags = np.load(os.path.join(subpath, flag_name))["arr_0"]
                gt_name = "gt.npz"
                gts = np.load(os.path.join(subpath, gt_name))["data"]
                gt = gts[traj_num, :, joint_num]
                valid_flag = valid_flags[traj_num, :]

        # print("id", id)
        # mean_name = "pred_mean_" + str(id) + ".npz"
        # pred_mu = np.load(os.path.join(subpath, mean_name))['arr_0']
        if subdir != "Transformer" and subdir != "Transformer-2" and subdir != "AR-Transformer" and subdir != "Multi-Transformer" and subdir != "AR" and subdir != "Multi":
            mean_name = "pred_mean_" + str(id) + ".npz"
            pred_mu = np.load(os.path.join(subpath, mean_name))['arr_0']
            if var_plot:
                std_name = "pred_var_" + str(id) + ".npz"
                pred_std = np.load(os.path.join(subpath, std_name))['arr_0']

            else:
                pred_std = None
        else:
            mean_name = "pred_mean_" + str(id) + ".npz"
            pred_mu = np.load(os.path.join(subpath, mean_name))["data"]
            std_name = "pred_var_" + str(id) + ".npz"
            if var_plot:
                pred_std = None
                # pred_std = np.load(os.path.join(subpath, std_name))["data"]
                # normalizer_name = "normalizer.npz"
                # ## load pickle file
            else:
                pred_std = None
            pred_std = None
            # print(os.path.join(subpath, normalizer_name))
            # normalizer = np.load(os.path.join(subpath, normalizer_name))["data"]
        # with open("/home/vshaj/CLAS/DP-SSM-v2/experiments/mobileRobot/conf_dp/data/SinMixNeuripsData.pkl",
        #           'rb') as f:
        #     data_dict = pickle.load(f)
        #     normalizer = data_dict['normalizer']
        # if pred_std is not None:
        #     pred_std = denorm_var(pred_std, normalizer)
        # pred_mu = denorm(pred_mu, normalizer)
        # gt = denorm(gt, normalizer)

        pred_mu = pred_mu[traj_num,:,joint_num]

        if var_plot and pred_std is not None:
            pred_std = pred_std[traj_num,:,joint_num]

        ## print shapes
        print("gt.shape", gt.shape)
        print("pred_mu.shape", pred_mu.shape)
        if pred_std is not None:
            print("pred_std.shape", pred_std.shape)
        print("valid_flag.shape", valid_flag.shape)

        ### select len(valid_flag) = episode_length
        select = len(gt) -len(valid_flag)
        gt = gt[select:]
        pred_mu = pred_mu[select:]
        if pred_std is not None:
            pred_std = pred_std[select:]

        print("gt.shape", gt.shape)
        print("pred_mu.shape", pred_mu.shape)
        if pred_std is not None:
            print("pred_std.shape", pred_std.shape)
        print("valid_flag.shape", valid_flag.shape)

        if subdir == "RKN-LL-.65" or subdir == "RKN-LL-new3":
            label_name = "RKN"
        elif subdir == "HiP-RSSM-LL":
            label_name = "HiP-RSSM"
        elif subdir == "lstm":
            label_name = "LSTM"
        elif subdir == "gru":
            label_name = "GRU"
        elif subdir == "MTS3-ActStateAbs":
            label_name = "MTS3"
        elif subdir == "AR":
            label_name = "AR-Transformer"
        elif subdir == "Multi":
            label_name = "Multi-Transformer"
        else:
            label_name = subdir



        with plt.rc_context(bundles.neurips2022()):
            ax.plot(gt, label='Ground Truth', color='black')
            ### squeeze valid flag
            valid_flag = valid_flag.squeeze()
            print("gt_dim.shape", gt[0:10], np.logical_not(valid_flag))

            if valid_flag is not None:
                if exp_name == "Neurips-Kitchen-D4RLnorm":
                    ax.scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],
                                gt[np.logical_not(valid_flag)], facecolor='red', s=10, label="Masked")
                else:
                    ax.scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],
                                    gt[np.logical_not(valid_flag)], facecolor='red', s=5, label="Masked")

            ax.plot(pred_mu, label="Predictions")

            if pred_std is not None:
                print(".........................................................................")
                print(pred_std)
                print(gt.shape, pred_mu.shape, pred_std.shape)
                ax.fill_between(np.arange(len(gt)), pred_mu - pred_std,
                                        pred_mu + pred_std, alpha=0.2, color='grey')

            ## title as label name
            #ax.set_title(label_name)
            ax.text(.11, .8, label_name,
                    horizontalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            if subdir=="RKN-LL" or subdir=="RKN-LL-new3" or subdir=="RKN-LL-.65":
                if exp_name!="Neurips-Kitchen-D4RLnorm_plots":
                    ##set ylimit as max and min of gt
                    max_gt = np.max(gt)
                    #min_gt = np.min(gt)
                    ax.set_ylim(-1.0, 1.0)

            ## show label
            ax.legend(loc='lower right', fontsize=8)


    return plt


def main():


    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10, 3.5))
    # ### for exp_name=Neurips-Kitchen-D4RLnorm_plots
    # plotAlgo(ax2, algo="AR", exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=75,
    #          traj_num=40, joint_num=0, var_plot=False, show=norm)
    # plotAlgo(ax3, algo="Multi", exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=75,
    #          traj_num=40, joint_num=0, var_plot=False, show=norm)
    # plotAlgo(ax1, algo="MTS3-ActStateAbs", exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=75,
    #          traj_num=40, joint_num=0, var_plot=False, show=norm)
    # plotAlgo(ax4, algo="lstm", exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=75, traj_num=40,
    #          joint_num=0, var_plot=False, show=norm)
    # plt.show()
    

    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8, 15))
    num =5
    plotAlgo(ax[0,0], algo="MTS3-ActC", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
                traj_num=num, joint_num=0, var_plot=False, show=norm)
    plotAlgo(ax[1,0], algo="MTS3-ActC", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
                traj_num=num, joint_num=1, var_plot=False, show=norm)
    plotAlgo(ax[2,0], algo="MTS3-ActC", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
                traj_num=num, joint_num=2, var_plot=False, show=norm)
    plotAlgo(ax[3,0], algo="MTS3-ActC", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
                traj_num=num, joint_num=3, var_plot=False, show=norm)
    plotAlgo(ax[4,0], algo="MTS3-ActC", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
                traj_num=num, joint_num=4, var_plot=False, show=norm)

    #plt.show()

    #fig, ax = plt.subplots(nrows=5, figsize=(8, 15))
    plotAlgo(ax[0,1], algo="MTS3-ActC-NoI", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
             traj_num=num, joint_num=0, var_plot=False, show=norm)
    plotAlgo(ax[1,1], algo="MTS3-ActC-NoI", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
             traj_num=num, joint_num=1, var_plot=False, show=norm)
    plotAlgo(ax[2,1], algo="MTS3-ActC-NoI", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
             traj_num=num, joint_num=2, var_plot=False, show=norm)
    plotAlgo(ax[3,1], algo="MTS3-ActC-NoI", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
             traj_num=num, joint_num=3, var_plot=False, show=norm)
    plotAlgo(ax[4,1], algo="MTS3-ActC-NoI", exp_name='Test-Mobile-NewCodenorm_plots', step_name=750, episode_length=75,
             traj_num=num, joint_num=4, var_plot=False, show=norm)

    plt.show()
    ## save fig as pdf
    #fig.savefig('Neurips-hydraulicsnorm_plots.pdf', bbox_inches='tight')

    
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3.5))
    #fig.tight_layout()
    #fig, ax1 = plt.subplots(ncols=1)
    # TO DO pass a list of joints as arguments... Currently if you want
    # to do multi joint training change line 61 directly eg: joints = [0,1,2,3,4,5,6]
    #plotCompare(exp_name='Baselines-240Hz-SinLongNextnorm_plots', step_name=10, episode_length=75, traj_num=12, show=norm)
    #for i in range(100):
    #    print("i", i)
    #plotCompare(exp_name='Neurips-hydraulicsnorm_plots', step_name=40, episode_length=30, traj_num=4, var_plot=False, show=norm)

    #plotCompare(exp_name='Neurips-SinMixLongNextnorm_plots', step_name=10, episode_length=75, traj_num=4, var_plot=False, show=norm)

    #plotMultiStepRMSE(ax2, "Mobile Robot", exp_name='Neurips-SinMixLongNextnorm_plots', step_name=10, episode_length=75, max_episodes=11, show=norm) ## this one
    # plotMultiStepRMSE(ax2, "MTS3 Variants on Mobile Robot", exp_name='Neurips-Ablation-SinMixnorm_plots', step_name=10, episode_length=75,
    #                   max_episodes=11, show=norm)  ## this one

    #plotMultiStepNLL(exp_name='Neurips-SinMixLongNextnorm_plots', step_name=10, episode_length=75, max_episodes=10,
    #               show=norm)
    #plotMultiStepRMSE(exp_name='Baselines-240Hz-SinLongNextnorm_plots', step_name=10, episode_length=75,
    #                max_episodes=10,
    #                show=norm)
    #plotMultiStepNLL(exp_name='Baselines-240Hz-SinLongNextnorm_plots', step_name=10, episode_length=75, max_episodes=10,
    #                    show=norm)
    #plotMultiStepRMSE(exp_name='Hydraulics-Neurips-Baselinesnorm_plots', step_name=40, episode_length=30, max_episodes=45,
    #                 show=norm)
    #plotCompare(exp_name='Baselines-Hydr-Next0norm_plots', step_name=40, episode_length=30, traj_num=500,
    #            show=norm)
    #plotMultiStepRMSE(ax1, title="HalfCheetah", exp_name='Neurips-Cheetah-D4RLnorm_plots', step_name=20, episode_length=15, max_episodes=24, show=norm) ## this one
    #plotMultiStepNLL(exp_name='Neurips-Cheetah-D4RLnorm_plots', step_name=20, episode_length=15, max_episodes=24, show=norm)
    #plotMultiStepRMSE(ax3, title="Kitchen", exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=15, max_episodes=12, show=norm) ## this one
    #plotMultiStepNLL(exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=15, max_episodes=12, show=norm)
    #plotMultiStepRMSE(ax2, title="Medium Maze", exp_name='Neurips1-medium-D4RLnorm_plots', step_name=13, episode_length=30, max_episodes=14, show=norm) ## this one
    #plotMultiStepNLL(exp_name='Neurips-medium-D4RLnorm_plots', step_name=13, episode_length=30, max_episodes=14,
    #                  show=norm)
    #plotMultiStepRMSE(ax2, title="AntMaze", exp_name='Neurips-UmazeAnt-D4RLnorm_plots', step_name=20, episode_length=15, max_episodes=22, show=norm)
    #plotMultiStepRMSE(ax4, title="U Maze", exp_name='Neurips-Umaze-D4RLnorm_plots', step_name=3, episode_length=30,
    #                  max_episodes=4, show=norm)  ## this one

    #plotMultiStepRMSE(ax2, title="U Maze", exp_name='Neurips-Umaze-D4RLnorm_plots', step_name=3, episode_length=30, max_episodes=4, show=norm) ## this one
    #plotMultiStepNLL(exp_name='Neurips-Umaze-D4RLnorm_plots', step_name=3, episode_length=30, max_episodes=4,
    #                 show=norm)

    #plotMultiStepRMSE(ax1, "MTS3 Variants on Excavator", exp_name='Neurips-hydraulicsnorm_plots', step_name=40, episode_length=30, max_episodes=43, show=norm)
    #plotMultiStepRMSE(ax2, "MTS3 Variants on Excavator", exp_name='Neurips-hydraulicsnorm_plots', step_name=40,
    #                 episode_length=30, max_episodes=43, show=norm)
    #plotMultiStepRMSE(ax2, "MTS3 Variants - SinMix", exp_name='Neurips-hydraulicsnorm_plots', step_name=40, episode_length=30,
     #                 max_episodes=43, show=norm)
    #plotMultiStepRMSE(ax1, title="Panda", exp_name='Neurips-Pandanorm_plots', step_name=6, episode_length=30, max_episodes=7, show=norm)  ## this one
    #plotMultiStepNLL(exp_name='Neurips-hydraulicsnorm_plots', step_name=40, episode_length=30, max_episodes=43,
    #                 show=norm)
    #plotMultiStepNLL(exp_name='Baselines-Hydr-Next0norm_plots', step_name=40, episode_length=30, max_episodes=45,
    #                  show=norm)
    #fig.text(0.5, 0.04, 'Seconds', ha='center', va='center')




    #######......................................................................................................
    #fig.text(0.06, 0.6, 'RMSE', ha='center', va='center', rotation='vertical', fontsize=16)
    #handles, labels = ax1.get_legend_handles_labels()
    #print(labels) #['Multi-Transformer', 'LSTM', 'GRU', 'MTS3', 'AR-Transformer', 'HiP-RSSM', 'RKN']
    #print(handles)
    #create order ['MTS3', 'LSTM', 'GRU',  'RKN', 'HiP-RSSM', 'AR-Transformer', 'Multi-Transformer']
    #labels = [labels[3], labels[1], labels[2], labels[6], labels[5], labels[4], labels[0]]
    #handles = [handles[3], handles[1], handles[2], handles[6], handles[5], handles[4], handles[0]]
    #fig.legend( handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.04), fontsize=14)
    #folder_name = "/home/vshaj/CLAS/DP-SSM-v2/experiments/output"
    #fig.subplots_adjust(bottom=0.3)
    #plt.savefig(folder_name + "/ablations.pdf")
    #plt.close()
    #plt.show()



if __name__ == '__main__':
    main()