##TODO: config setting
import sys
sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os

import numpy as np
import torch
import wandb
import pickle
import json

from dataFolder.mobileDataDpssm_v1 import metaMobileData
from agent.worldModels import MTS3
from agent.Learn.repre_learn_mts3 import Learn
from inference import dprssm_inference
from utils.metrics import naive_baseline
from utils.dataProcess import split_k_m, denorm, denorm_var
from utils.metrics import root_mean_squared, joint_rmse, gaussian_nll
from utils.latentVis import plot_clustering, plot_clustering_1d
from hydra.utils import get_original_cwd, to_absolute_path
from utils.plotTrajectory import plotImputation, plotMbrl, plotLongTerm

nn = torch.nn

class Experiment():
    def __init__(self, cfg):
        self.global_cfg = cfg 
        self._data_train_cfg = self.global_cfg.data.train
        self._data_test_cfg = self.global_cfg.data.test
        # 'next_state' - if to trian directly on the  next states
        assert self._data_train_cfg.tar_type == self._data_test_cfg.tar_type #"Train and Test Target Types are same"
        torch.cuda.empty_cache()
        
        print("Terrain Type.........", self._data_train_cfg.terrain)

    def _reshape_data(data):
        ## reshape the data by flattening the second and third dimension
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2], -1)
        return data
    
    def _load_save_train_test_data(self, dataLoaderClass):
        ### Load the data from pickle or generate the data and save it in pickle
        if self._data_train_cfg.load:
            with open(get_original_cwd() + self._data_train_cfg , 'rb') as f:
                data = pickle.load(f)
            with open(get_original_cwd() + self._data_test_cfg.save_path, 'rb') as f:
                data_test = pickle.load(f)
            print("..........Data Loaded from Pickle...........")
        else:
            data = dataLoaderClass(self._data_train_cfg)
            data_test = dataLoaderClass(self._data_test_cfg)

            if self._data_train_cfg.save:
                with open(get_original_cwd() + self._data_train_cfg.save_path, 'wb') as f:
                    pickle.dump(data, f)
                with open(get_original_cwd() + self._data_test_cfg.save_path, 'wb') as f:
                    pickle.dump(data_test, f)
                print("..........Data Saved To Pickle...........")
        return data, data_test
    
    def _convert_to_tensor(self, data):
        train_windows, test_windows = data.train_windows, data.test_windows
        train_windows, test_windows = data.train_windows, data.test_windows

        train_targets = torch.from_numpy(train_windows['target']).float()
        test_targets = torch.from_numpy(test_windows['target']).float()

        train_obs = torch.from_numpy(train_windows['obs']).float()
        test_obs = torch.from_numpy(test_windows['obs']).float()

        train_act = torch.from_numpy(train_windows['act']).float()
        test_act = torch.from_numpy(test_windows['act']).float()

        print("Fraction of Valid Train Observations:",
            np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
        print("Fraction of Valid Test Observations:",
            np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

        return train_obs, train_act, train_targets, test_obs, test_act, test_targets
    
    
    def _get_data_set():
        ### define in the child class depending on the dataset
        raise NotImplementedError

        
    def _wandb_init(self):
        ## Convert Omega Config to Wandb Config (letting wandb know of the config for current run)
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        expName = cfg.wandb.exp_name + cfg.learn.name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"
        ## Initializing wandb object and sweep object
        if cfg.wandb.log:
            wandb.login(key="55cdb950375b3a8f9ca3d6846e7b2f90b62547f8", relogin=True)
        wandb_run = wandb.init(config=config_dict, project=cfg.wandb.project_name, name=expName,
                                    mode=mode)  # wandb object has a set of configs associated with it as well 
        return wandb_run
        

    def _train_world_model(self, train_obs, train_act, train_targets, test_obs, test_act, test_targets):
        ##### Define WandB Stuffs
        wandb_run = self._wandb_init()

        ### Setting save_path for the model based on the wandb_run id
        if cfg.learn.model.load == False:
            save_path = get_original_cwd() + '/experiments/saved_models/' + wandb_run.id + '.ckpt'
        else:
            save_path = get_original_cwd() + '/experiments/saved_models/' + cfg.learn.model.id + '.ckpt'

        ### Model Initialize, Train and Inference Modules

        world_model = MTS3()
        dp_learn = dprssm_trainer.Learn(dp_model, loss=cfg.learn.loss, config=cfg, run=wandb_run,
                                            log=cfg.wandb['log'])

        if cfg.learn.model.load == False:
            #### Train the Model
            print(test_obs_valid.shape, test_task_valid.shape)
            dp_learn.train(train_obs, train_act, train_targets[:,1:], train_task_idx, cfg.learn.epochs, cfg.learn.batch_size,
                            test_obs, test_act,
                            test_targets[:,1:], test_task_idx)
            
        return dp_model, wandb_run, save_path
            

    def _evaluate_world_model(self, test_obs, test_act, test_targets, test_task_idx, dp_model, wandb_run, save_path):

        ##### Inference Module
        dp_infer = dprssm_inference.Infer(dp_model, normalizer=data.normalizer, config=cfg, run=wandb_run,
                                            log=cfg.wandb['log'])

        ##### Load best model
        dp_model.load_state_dict(torch.load(save_path))
        print('>>>>>>>>>>Loaded The Model From Local Folder<<<<<<<<<<<<<<<<<<<')

        ##### Inference From Loaded Model for imputation
        pred_mean, pred_var, gt, obs_valid, cur_obs, l_prior, l_post, task_labels = dp_infer.predict(test_obs, test_act,
                                                                                test_targets, test_task_idx,
                                                                                batch_size=1000, tar=tar_type)
        namexp = str(impu) + "/Imp"

        #plotImputation(gt, obs_valid, pred_mean, pred_var, wandb_run, l_prior, l_post, task_labels,  exp_name=namexp)

        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data.normalizer,
                                                                tar="observations", denorma=True)
        wandb_run.summary['rmse_denorma_next_state'] = rmse_next_state


        ### Calculate the RMSE for imputation normalized
        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data.normalizer,
                                                                tar="observations", denorma=False)
        wandb_run.summary['nrmse_next_state'] = rmse_next_state

        joint_rmse_next_state = joint_rmse(pred_mean, gt, data.normalizer,
                                            tar="observations", denorma=False)
        for joint in range(joint_rmse_next_state.shape[-1]):
            wandb_run.summary['nrmse_next_state' + "_joint_" + str(joint)] = joint_rmse_next_state[joint]

        print("Root mean square Error is:", rmse_next_state)

        ### Multi Step Inference From Loaded Model
        ### TODO: Create a lot more test sequences

        for step in range(0, test_obs.shape[1]-1):
            if step in [0, 1, int(test_obs.shape[-1] / 2), test_obs.shape[-1] - 1] or step % 5 == 0:
                pred_mean, pred_var, gt, obs_valid, cur_obs, l_prior, l_post, task_labels = dp_infer.predict_multistep(test_obs, test_act,
                                                                                                    test_targets,
                                                                                                    test_task_idx,
                                                                                                    multistep=step,
                                                                                                    batch_size=1000,
                                                                                                    tar=tar_type)

                if test_obs.shape[1]-2 == step:
                    multiStepNew(gt, pred_mean, pred_var, data)
                ### Denormalize the predictions and ground truth
                pred_mean_denorm = denorm(pred_mean, data.normalizer, tar_type=tar_type); pred_var_denorm = denorm_var(pred_var, data.normalizer, tar_type=tar_type); gt_denorm = denorm(gt, data.normalizer, tar_type=tar_type)
                ### Plot the normalized predictions
                namexp = cfg.wandb.project_name + "norm_plots/" + str(step) + "/" + cfg.wandb.exp_name
                plotImputation(gt, obs_valid, pred_mean, pred_var, wandb_run, l_prior, l_post, task_labels, exp_name=namexp)



                #######:::::::::::::::::::Calculate the RMSE and NLL for multistep normalized:::::::::::::::::::::::::::::::::::::
                pred_mean_multistep = pred_mean[:, -data.episode_length:, :]
                pred_var_multistep = pred_var[:, -data.episode_length:, :]
                gt_multistep = gt[:, -data.episode_length:, :]
                rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                        data.normalizer,
                                                                        tar="observations", denorma=False)
                nll_next_state, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep,
                                                        data.normalizer,
                                                        tar="observations",
                                                        denorma=False)
                wandb_run.summary['norm_nll_multi_step_' + str(step)] = nll_next_state
                wandb_run.summary['nrmse_multistep' + str(step)] = rmse_next_state

                print("Multi Step NRMSE - Step (x.3s) -" + str(step), rmse_next_state)

                #########:::::::::::::::::::Calculate denoramalized RMSE and NLL for multi step ahead predictions:::::::::::::::::::
                rmse_next_state, _, _ = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                        data.normalizer,
                                                                        tar="observations", denorma=True)
                nll_next_state, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep, data.normalizer, tar="observations",
                                        denorma=True)
                namexp = cfg.wandb.project_name + "true_plots/" + str(step) + "/" + cfg.wandb.exp_name
                plotImputation(gt_denorm, obs_valid, pred_mean_denorm, pred_var_denorm, wandb_run, l_prior, l_post, task_labels, exp_name=namexp)
                wandb_run.summary['rmse_multi_step_' + str(step)] = rmse_next_state
                wandb_run.summary['nll_multi_step_' + str(step)] = nll_next_state

                ## Logging joint wise denormalized multi step ahead predictions
                joint_rmse_next_state = joint_rmse(pred_mean, gt, data.normalizer,
                                                    tar="observations", denorma=True)
                for joint in range(joint_rmse_next_state.shape[-1]):
                    wandb_run.summary['rmse_multistep_' + str(step) + "_joint_" + str(joint)] = joint_rmse_next_state[
                        joint]



                def multiStepNew(gt, pred_mu, pred_std, data):
                    ## A different way of calculating the multi step ahead rmse
                    ## You first do a very long multistep ahead prediction
                    ##  and you trucate the predicted sequence at definite steps and calculate different step ahead rmse
                    ## makes it more farer for hiprssm
                    for step in [gt.shape[1] / 10, gt.shape[1] / 7, gt.shape[1] / 5, gt.shape[1] / 3, gt.shape[1] / 2,
                                    gt.shape[1]]:
                        step = int(step)
                        episode_length = data.episode_length

                        pred_mean_multistep = pred_mu[:, episode_length:step, :]
                        pred_var_multistep = pred_std[:, episode_length:step, :]
                        gt_multistep = gt[:, episode_length:step, :]
                        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                                    data.normalizer,
                                                                                tar="observations", denorma=True)
                        nll, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep)
                        wandb_run.summary['new_rmse_true_' + str(step)] = rmse_next_state
                        wandb_run.summary['new_nll_true_' + str(step)] = nll

                        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                                data.normalizer,
                                                                                tar="observations", denorma=False)
                        nll, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep)
                        wandb_run.summary['new_rmse_norm_' + str(step)] = rmse_next_state
                        wandb_run.summary['new_nll_norm_' + str(step)] = nll








def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()