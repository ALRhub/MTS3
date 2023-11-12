import sys

sys.path.append('.')
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.metrics import root_mean_squared, joint_rmse, gaussian_nll
from tueplots import bundles



def plotJoints(fig, axs, gt, pred_mu, pred_std, valid_flag, label_name, use_bundle=True, firstPlot=True):
    dim = 0
    print("gt.shape", gt.shape)
    print("pred_mu.shape", pred_mu.shape)
    #print("pred_std.shape", pred_std.shape)
    print("valid_flag.shape", valid_flag.shape)
    if use_bundle:
        with plt.rc_context(bundles.neurips2022()):
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    if dim >= gt.shape[-1]:
                        break
                    gt_dim = gt[:, dim]
                    pred_mu_dim = pred_mu[:, dim]
                    if pred_std is not None:
                        pred_std_dim = pred_std[:, dim]
                    dim = dim + 1
                    if firstPlot:
                        axs[i, j].plot(gt_dim, label='gt', color='black')
                        ### squeeze valid flag
                        valid_flag = valid_flag.squeeze()
                        print("gt_dim.shape", gt_dim[0:10],np.logical_not(valid_flag))

                        if valid_flag is not None:
                            axs[i, j].scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],
                                                gt_dim[np.logical_not(valid_flag)], facecolor='red', s=5)

                    axs[i, j].plot(pred_mu_dim, label=label_name)
                    axs[i, j].set_title('Joint ' + str(dim))
                    ## set legend lower left
                    axs[i, j].legend(loc='lower left')

                    if pred_std is not None:
                        print(".........................................................................")
                        print(pred_std)
                        print(gt.shape, pred_mu.shape, pred_std.shape)
                        axs[i, j].fill_between(np.arange(len(gt)), pred_mu_dim - pred_std_dim,
                                                pred_mu_dim + pred_std_dim, alpha=0.2, color='grey')
            firstPlot = False
    else:
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                if dim >= gt.shape[-1]:
                    break
                gt_dim = gt[:, dim]
                pred_mu_dim = pred_mu[:, dim]
                if pred_std is not None:
                    pred_std_dim = pred_std[:, dim]
                dim = dim + 1
                if firstPlot:
                    axs[i, j].plot(gt_dim, label='gt', color='black')

                    if valid_flag is not None:
                        axs[i, j].scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],
                                            gt_dim[np.logical_not(valid_flag)], facecolor='red', s=10)
                axs[i, j].plot(pred_mu_dim, label=label_name)
                axs[i, j].set_title('Joint ' + str(dim))
                ## set legend lower left
                axs[i, j].legend(loc='lower left')

                if pred_std is not None:
                    axs[i, j].fill_between(np.arange(len(gt)), pred_mu_dim - pred_std_dim, pred_mu_dim + pred_std_dim,
                                            alpha=0.2, color='grey')

    return fig, axs


def plotMultiStepWindowedRMSE(ax, title, exp_name, target_steps, context_steps, window_len, data_frequency, show=True):
    """
    :param ax: axis object 
    :param title: plot title (usually algo name)
    :param exp_name: name of the experiment
    :param target_steps: number of masked steps (multistep ahead predictions)
    :param context_steps: number of context steps (initial context shown)
    :param window_len: length of the window to calculate moving average
    :param data_frequency: frequency of the data
    :param show: whether to show the plot or not
    :return: 
    """
    folder_name = os.getcwd() + "/logs/output/plots/" + exp_name + '/' + str(target_steps)
    ## loop over all the subfolders in a folder path
    firstPlot = True
    with plt.rc_context(bundles.neurips2022()):
        for subdir in os.listdir(folder_name):
            ### covert to lower case
            if subdir.lower() in ["acrkn", "hip-rssm", "mts3", "lstm", "gru","mts3-noi","mts3V2","mts3f","mts3-noif" ]:
                print("---------------------------------------------------------",subdir)
                id = None
                subpath = os.path.join(folder_name, subdir)
                file_rmse_list = []
                id_list = []
                ## loop over files in subpath
                for file in os.listdir(subpath):
                    print(file)
                    ## if file ends with ".npz"
                    if not file.endswith(".npz"):
                        continue
                    ## if file not "gt.npz"
                    if file == "gt.npz":
                        continue
                    else:
                        id = file.split("_")[-1].split(".")[0]
                        if id in id_list:
                            continue
                        id_list.append(id)
                    print("id", id)

                    if subdir != "Transformer" and subdir != "Transformer-2" and subdir != "AR-Transformer" and subdir != "Multi-Transformer":
                        mean_name = "pred_mean_" + str(id) + ".npz"
                        pred_mu = np.load(os.path.join(subpath, mean_name))['arr_0']
                        std_name = "pred_var_" + str(id) + ".npz"
                        pred_std = np.load(os.path.join(subpath, std_name))['arr_0']
                        flag_name = "valid_flags_" + str(id) + ".npz"
                        valid_flag = np.load(os.path.join(subpath, flag_name))['arr_0']
                        gt_name = "gt.npz"
                        gts = np.load(os.path.join(subpath, gt_name))['arr_0']
                    else:
                        mean_name = "pred_mean_" + str(id) + ".npz"
                        gt_name = "gt.npz"
                        pred_mu = np.load(os.path.join(subpath, file))["data"]
                        gts = np.load(os.path.join(subpath, gt_name))["data"]

                    print("ground truth shape", gts.shape)
                    rmse_list = []
                    second_list = []

                    pred_mean_target = pred_mu[:, -target_steps:, :]
                    print("pred_mean_masked", pred_mean_target.shape)
                    gts_target = gts[:, -target_steps:, :]
                    print("gts_masked", gts_target.shape)

                    ### Calculating Multi-Step RMSE using moving average
                    for num_steps in [target_steps / 50, target_steps / 9, target_steps / 8,
                                target_steps / 5,  target_steps / 3,   target_steps / 2, target_steps]:
                    #for step in [target_steps / 9, target_steps / 6,
                    #            target_steps / 3, target_steps / 2]:
                        num_steps = int(num_steps)
                        has_large_value = False
                        print("step", num_steps)
                        second_list.append(num_steps/data_frequency)
                        if num_steps < window_len:
                            history = num_steps
                        else:
                            history = window_len


                        pred_mean_multistep = pred_mean_target[:, num_steps-history:num_steps, :]
                        gt_multistep = gts_target[:, num_steps-history:num_steps, :]

                        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                                None, tar="observations", denorma=False)

                        rmse_list.append(rmse_next_state)
                    file_rmse_list.append(rmse_list)
                    print("rmse_list", rmse_list)

                if has_large_value:
                    continue

                print("file_rmse_list", file_rmse_list)

                rmse_list = np.mean(file_rmse_list, axis=0)
                std_list = np.std(file_rmse_list, axis=0)

                ## plot the rmse_list
                # print(second_list, rmse_list)
                if subdir == "RKN-LL" or subdir == "RKN-LL-new3":
                    label_name = "RKN"
                elif subdir == "HiP-RSSM-LL" or subdir == "HiP-RSSM-LL-new3":
                    label_name = "HiP-RSSM"
                elif subdir == "lstm":
                    label_name = "LSTM"
                elif subdir == "gru":
                    label_name = "GRU"
                elif subdir == "MTS3-ActStateAbs":
                    label_name = "MTS3"
                elif subdir == "MTS3-StateAbs2" or subdir == "MTS3-StateAbs":
                    label_name = "No Action Abstraction"
                elif subdir == "MTS3-NoMemC" or subdir=="MTS3-NoMem":
                    label_name = "No Memory"
                elif subdir == "MTS3-noimp":
                    label_name = "No Imputation"
                elif subdir == "MTS3":
                    label_name = "WIth-I"
                else:
                    label_name = subdir

                #ax.plot(second_list, rmse_list, linestyle=line, color='red', label=label_name, linewidth=3)
                ax.plot(second_list, rmse_list, label=label_name, linewidth=3)

                #clip std_list between -1 and 1
                #std_list = np.clip(std_list, -1, 0.5)
                ax.fill_between(second_list, rmse_list - std_list, rmse_list + std_list, alpha=0.2)
                ### clip between min and max
                if label_name.lower()=="mts3-noi":
                    if "Panda" in title:
                        min = 0.05
                    else:
                        min = np.min(rmse_list-std_list)
                    if title == "Medium Maze":
                        max = 1.4**np.max(rmse_list + std_list)
                    else:
                        max = np.max(rmse_list + std_list)
                    ax.set_ylim([min, max])
                ### log scale
                #ax.set_yscale('log')

                #plt.xlim([0, 0.5])

        ax.legend(loc='upper right', fontsize=14)
        ax.set_xlabel("Seconds", fontsize="14")
        ## set tile
        ax.set_title(title, fontsize="16")
        #ax.xlabel("Seconds")
        #ax.ylabel("RMSE")
        if show == True:
            return plt
            #plt.show()
            #plt.close()
        else:
            plt.savefig(folder_name + "/rmse_" + exp_name + ".pdf")
            plt.close()




def plotMultiStepWindowedNLL(ax, title, exp_name, target_steps, context_steps, window_len, data_frequency, show=True):
    """
    :param ax: axis object
    :param title: plot title (usually algo name)
    :param exp_name: name of the experiment
    :param target_steps: number of masked steps (multistep ahead predictions)
    :param context_steps: number of context steps (initial context shown)
    :param window_len: length of the window to calculate moving average
    :param data_frequency: frequency of the data
    :param show: whether to show the plot or not
    :return:
    """
    folder_name = os.getcwd() + "/logs/output/plots/" + exp_name + '/' + str(target_steps)
    ## loop over all the subfolders in a folder path
    firstPlot = True
    fig, ax = plt.subplots()
    with plt.rc_context(bundles.jmlr2001()):
        for subdir in os.listdir(folder_name):
            if subdir.lower() in ["mts3","acrkn","hip-rssm","mts3-noi","gru","lstm" ]:
                id = None
                print(subdir)
                subpath = os.path.join(folder_name, subdir)
                file_nll_list = []
                id_list = []
                ## loop over files in subpath
                for file in os.listdir(subpath):
                    print(file)
                    ## if file not "gt.npz"
                    if file == "gt.npz":
                        continue
                    else:
                        id = file.split("_")[-1].split(".")[0]
                        if id in id_list:
                            continue
                        id_list.append(id)
                    print("id", id)
                    mean_name = "pred_mean_" + str(id) + ".npz"
                    pred_mu = np.load(os.path.join(subpath, mean_name))['arr_0']
                    if subdir != "Transformer" and subdir != "Transformer-2" and subdir != "AR-Transformer" and subdir!= "Multi-Transformer":
                        std_name = "pred_var_" + str(id) + ".npz"
                        pred_std = np.load(os.path.join(subpath, std_name))['arr_0']
                        flag_name = "valid_flags_" + str(id) + ".npz"
                        valid_flag = np.load(os.path.join(subpath, flag_name))['arr_0']
                    gt_name = "gt.npz"
                    gts = np.load(os.path.join(subpath, gt_name))['arr_0']
                    print("ground truth shape", gts.shape)
                    nll_list = []
                    second_list = []

                    pred_mean_target = pred_mu[:, -target_steps:, :]
                    print("pred_mean_masked", pred_mean_target.shape)
                    pred_std_target = pred_std[:, -target_steps:, :]
                    print("pred_std_masked", pred_std_target.shape)
                    gts_target = gts[:, -target_steps:, :]
                    print("gts_masked", gts_target.shape)

                    ### Calculating Multi-Step RMSE using moving average
                    for num_steps in [target_steps / 50, target_steps / 9, target_steps / 8, target_steps / 7,
                                        target_steps / 6, target_steps / 5, target_steps / 4, target_steps / 3, target_steps / 2,
                                        target_steps / 1.5, target_steps]:
                        # for step in [target_steps / 9, target_steps / 6,
                        #            target_steps / 3, target_steps / 2]:
                        has_large_value = False
                        num_steps = int(num_steps)
                        print("step", num_steps)
                        second_list.append(num_steps / data_frequency)
                        if num_steps < window_len:
                            history = num_steps
                        else:
                            history = window_len

                        if subdir is not "Transformer":
                            pred_mean_multistep = pred_mean_target[:, num_steps - history:num_steps, :]
                            pred_var_multistep = pred_std_target[:, num_steps - history:num_steps, :]
                            gt_multistep = gts_target[:, num_steps - history:num_steps, :]
                        else:
                            pred_mean_multistep = pred_mean_target[:, num_steps - history:num_steps, :]
                            pred_var_multistep = None
                            gt_multistep = gts_target[:, num_steps - history:num_steps, :]

                        nll_next_state, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep, None, tar="observations", denorma=False)
                        nll_list.append(nll_next_state)
                    file_nll_list.append(nll_list)
                    print("rmse_list", nll_list)

                print("file_rmse_list", file_nll_list)

                rmse_list = np.mean(file_nll_list, axis=0)
                std_list = np.std(file_nll_list, axis=0)

                ## plot the rmse_list
                if subdir == "RKN-LL":
                    label_name = "RKN"
                elif subdir == "HiP-RSSM-LL":
                    label_name = "HiP-RSSM"
                elif subdir == "lstm":
                    label_name = "LSTM"
                elif subdir == "gru":
                    label_name = "GRU"
                elif subdir == "MTS3-ActStateAbs":
                    label_name = "MTS3"
                elif subdir == "MTS3":
                    label_name = "With-I"
                else:
                    label_name = subdir
                ax.plot(second_list, rmse_list, label=label_name, linewidth=3)
                ax.fill_between(second_list, rmse_list - std_list, rmse_list + std_list, alpha=0.2)
                ### clip between min and max
                ax.set_ylim([-15, 1])
                ### log scale
                ax.legend(loc='upper right', fontsize=14)
                ax.set_xlabel("Seconds", fontsize="14")
                ## set tile
                ax.set_title(title, fontsize="16")
                #ax.set_yscale('log')
        if show == True:
            return plt
            #plt.show()
            #plt.close()
        else:
            plt.savefig(folder_name + "/nll_" + exp_name + ".pdf")
            plt.close()


def main():
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 3.5))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3.5))
    #fig.tight_layout()
    #fig, ax1 = plt.subplots(ncols=1)
    # TO DO pass a list of joints as arguments... Currently if you want
    # to do multi joint training change line 61 directly eg: joints = [0,1,2,3,4,5,6]
    #plotCompare(exp_name='Neurips-SinMixLongNexttrue_plots', step_name=10, episode_length=75, traj_num=10, show=True)
    #for i in range(100):
    #    print("i", i)
    #plotCompare(exp_name='Neurips-hydraulicsnorm_plots', step_name=40, episode_length=30, traj_num=4, var_plot=True, show=True)

    #plotCompare(exp_name='Neurips-SinMixLongNexttrue_plots', step_name=10, episode_length=75, traj_num=4, var_plot=True, show=True)
    #plotMultiStepRMSE(ax1, title="Adroit", exp_name='Neurips-Andro1-D4RLtrue_plots', step_name=2, episode_length=49, max_episodes=4, show=True) ## this one
    #plotMultiStepWindowedNLL(ax2, "Mobile Robot", exp_name='cameraMobiletrue_plots', context_steps=150, target_steps=750, window_len=150, data_frequency=250, show=True) ## this one
    #plotMultiStepWindowedRMSE(ax1, "Franka Kitchen", exp_name='cameraFrankaKitchentrue_plots', context_steps=30, target_steps=150, window_len=50, data_frequency=100, show=True)
    #plotMultiStepWindowedNLL(ax2, "Franka Kitchen", exp_name='cameraFrankaKitchentrue_plots', context_steps=30, target_steps=150, window_len=50, data_frequency=100, show=True)

    #plotMultiStepWindowedRMSE(ax1, "Mobile Robot", exp_name='cameraMobiletrue_plots', context_steps=150, target_steps=750, window_len=150, data_frequency=250, show=True)
    ### do this for maze2d
    plotMultiStepWindowedRMSE(ax1, "Medium Maze", exp_name='Test-Maze2d-NewCodetrue_plots', context_steps=60, target_steps=390, window_len=30, data_frequency=100, show=True)
    plotMultiStepWindowedNLL(ax2, "Medium Maze", exp_name='Test-Maze2d-NewCodetrue_plots', context_steps=60, target_steps=390, window_len=30, data_frequency=100, show=True)
    ### do the same witj halfcheetah
    #plotMultiStepWindowedRMSE(ax1, "HalfCheetah", exp_name='Test-HalfCheetah-NewCodetrue_plots', context_steps=60, target_steps=330, window_len=15, data_frequency=50, show=True)
    #plotMultiStepWindowedNLL(ax2, "HalfCheetah", exp_name='Test-HalfCheetah-NewCodenorm_plots', context_steps=60, target_steps=330, window_len=15, data_frequency=50, show=True)
    ### do the same with hydraulics
    #plotMultiStepWindowedRMSE(ax1, "Hydraulics", exp_name='Test-Hydraulics-NewCodetrue_plots', context_steps=150, target_steps=1290, window_len=30, data_frequency=100, show=True)
    #plotMultiStepWindowedNLL(ax2, "Hydraulics", exp_name='Test-Hydraulics-NewCodetrue_plots', context_steps=150, target_steps=1290, window_len=30, data_frequency=100, show=True)

    #plotCompare(exp_name='Baselines-Hydr-Next0norm_plots', step_name=40, episode_length=30, traj_num=500,
    #            show=True)
    #plotMultiStepRMSE(ax1, title="HalfCheetah", exp_name='Neurips-Cheetah-D4RLtrue_plots', step_name=20, episode_length=15, max_episodes=24, show=True) ## this one
    #plotMultiStepNLL(exp_name='Neurips-Cheetah-D4RLtrue_plots', step_name=20, episode_length=15, max_episodes=24, show=True)
    #plotMultiStepRMSE(ax3, title="Kitchen", exp_name='Neurips-Kitchen-D4RLtrue_plots', step_name=8, episode_length=15, max_episodes=12, show=True) ## this one
    #plotMultiStepNLL(exp_name='Neurips-Kitchen-D4RLtrue_plots', step_name=8, episode_length=15, max_episodes=12, show=True)
    #plotMultiStepRMSE(ax2, title="Medium Maze", exp_name='Neurips1-medium-D4RLtrue_plots', step_name=13, episode_length=30, max_episodes=14, show=True) ## this one
    #plotMultiStepNLL(exp_name='Neurips-medium-D4RLnorm_plots', step_name=13, episode_length=30, max_episodes=14,
    #                  show=True)
    #plotMultiStepRMSE(ax2, title="AntMaze", exp_name='Neurips-UmazeAnt-D4RLtrue_plots', step_name=20, episode_length=15, max_episodes=22, show=True)
    #plotMultiStepRMSE(ax4, title="U Maze", exp_name='Neurips-Umaze-D4RLtrue_plots', step_name=3, episode_length=30,
    #                  max_episodes=4, show=True)  ## this one

    #plotMultiStepRMSE(ax2, title="U Maze", exp_name='Neurips-Umaze-D4RLtrue_plots', step_name=3, episode_length=30, max_episodes=4, show=True) ## this one
    #plotMultiStepNLL(exp_name='Neurips-Umaze-D4RLtrue_plots', step_name=3, episode_length=30, max_episodes=4,
    #                 show=True)

    #plotMultiStepRMSE(ax1, "Excavator", exp_name='Neurips-hydraulicstrue_plots', step_name=40, episode_length=30, max_episodes=43, show=True)
    #plotMultiStepRMSE(ax2, "MTS3 Variants on Excavator", exp_name='Neurips-hydraulicstrue_plots', step_name=40,
        #             episode_length=30, max_episodes=43, show=True)
    #plotMultiStepRMSE(ax2, "MTS3 Variants - Excavator", exp_name='Neurips-hydraulicstrue_plots', step_name=40, episode_length=30,
    #                  max_episodes=43, show=True)
    #plotMultiStepRMSE(ax4, title="Panda", exp_name='Neurips-Pandatrue_plots', step_name=6, episode_length=30, max_episodes=7, show=True)  ## this one
    #plotMultiStepNLL(exp_name='Neurips-hydraulicstrue_plots', step_name=40, episode_length=30, max_episodes=43,
    #                 show=True)
    #plotMultiStepNLL(exp_name='Baselines-Hydr-Next0norm_plots', step_name=40, episode_length=30, max_episodes=45,
    #                  show=True)
    #fig.text(0.5, 0.04, 'Seconds', ha='center', va='center')

    print("Panda..............")
    #metricsNll(exp_name='Neurips-Pandanorm_plots', step_name=6, episode_length=30,max_episodes=7, show=True)  ## this one
    print("U Maze..............")
    #metricsNll(exp_name='Neurips-Umaze-D4RLnorm_plots', step_name=3, episode_length=30, max_episodes=4,
    #                show=True)
    print("Medium Maze..............")
    #metricsNll(exp_name='Neurips1-medium-D4RLnorm_plots', step_name=13,
    #                 episode_length=30, max_episodes=14, var_plot=True, show=True)  ## this one
    #print("Medium Maze..............")
    #metricsNll(exp_name='Neurips-medium-D4RLnorm_plots', step_name=13, episode_length=30, max_episodes=14,
    #               show=True)
    print("Kitchen..............")
    #metricsNll(exp_name='Neurips-Kitchen-D4RLnorm_plots', step_name=8, episode_length=15, max_episodes=12, show=True)
    print("HalfCheetah..............")
    #metricsNll(exp_name='Neurips-Cheetah-D4RLnorm_plots', step_name=20, episode_length=15, max_episodes=24, show=True)
    print("Hydraulics..............")
    #metricsNll(exp_name='Neurips-hydraulicsnorm_plots', step_name=40, episode_length=30, max_episodes=43, show=True)
    print("SinMix..............")
    #metricsNll(exp_name='Neurips-SinMixLongNextnorm_plots', step_name=10, episode_length=75, max_episodes=11, show=True)




    ######......................................................................................................
    fig.text(0.1, 0.6, 'RMSE', ha='center', va='center', rotation='vertical', fontsize=16)
    handles, labels = ax1.get_legend_handles_labels()
    #print(labels) #['Multi-Transformer', 'LSTM', 'GRU', 'MTS3', 'AR-Transformer', 'HiP-RSSM', 'RKN']
    # print(handles)
    # #labels = ['MTS3',  'No Memory', 'No Action Abstraction', 'No Imputation']
    # #handles = [handles[2], handles[3], handles[0], handles[1]]
    # #create order ['MTS3', 'LSTM', 'GRU',  'RKN', 'HiP-RSSM', 'AR-Transformer', 'Multi-Transformer']
    # labels = [labels[3], labels[1], labels[2], labels[6], labels[5], labels[4], labels[0]]
    # handles = [handles[3], handles[1], handles[2], handles[6], handles[5], handles[4], handles[0]]
    fig.legend( handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.04), fontsize=14)
    # folder_name = "/home/vshaj/CLAS/DP-SSM-v2/experiments/output"
    fig.subplots_adjust(bottom=0.3)
    # plt.savefig(folder_name + "/d_rmse.pdf")
    # #plt.close()
    plt.show()



if __name__ == '__main__':
    main()