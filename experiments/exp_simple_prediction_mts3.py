##TODO: avoid convert to tensor here
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
from torch.nn.parallel import DataParallel

from agent.worldModels.MTS3Simple import MTS3Simple
from agent.Learn.repre_learn_mts3 import Learn
from agent.Infer.repre_infer_mts3 import Infer
from utils.dataProcess import split_k_m, denorm, denorm_var
from utils.metrics import root_mean_squared, joint_rmse, gaussian_nll
from hydra.utils import get_original_cwd, to_absolute_path
from utils.plotTrajectory import plotImputation, plotMbrl, plotLongTerm

nn = torch.nn

class Experiment():
    """
    Experiment class for training and testing the simple MTS3 model (unactuated time series model)"""
    def __init__(self, cfg):
        self.model_cfg = cfg.model
        self.learn_cfg = self.model_cfg.learn
        self._data_cfg = self.model_cfg.data
        # 'next_state' - if to trian directly on the  next states
        assert self._data_cfg.tar_type == self._data_cfg.tar_type #"Train and Test Target Types are same"
        torch.cuda.empty_cache()
        
        print("Terrain Type.........", self._data_cfg.terrain)

    def _reshape_data(self, data):
        ## reshape the data by flattening the second and third dimension
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2], -1)
        return data
    
    def _load_save_train_test_data(self, dataLoaderClass):
        ### Load the data from pickle or generate the data and save it in pickle
        if self._data_cfg.load:
            ## load the data from pickle and if not present download from the url
            if not os.path.exists(get_original_cwd() + self._data_cfg.save_path):
                print("..........Data Not Found...........Downloading from URL")
                ### download the data from url
                from urllib.request import urlretrieve
                urlretrieve(self._data_cfg.url, get_original_cwd() + self._data_cfg.save_path)
            else:
                print("..........Data Found...........Loading from Pickle")
            with open(get_original_cwd() + self._data_cfg.save_path, 'rb') as f:
                data = pickle.load(f)
            print("..........Data Loaded from Pickle...........")
        else:
            data = dataLoaderClass(self._data_cfg)

            if self._data_cfg.save:
                with open(get_original_cwd() + self._data_cfg.save_path, 'wb') as f:
                    pickle.dump(data, f)
                print("..........Data Saved To Pickle...........")
        return data
    
    def _convert_to_tensor_reshape(self, data):
        ### Convert data to tensor (maybe move this to dataLoaderClass)
        train_windows, test_windows = data.train_windows, data.test_windows

        train_targets = torch.from_numpy(train_windows['target']).float()
        train_targets = self._reshape_data(train_targets)
        test_targets = torch.from_numpy(test_windows['target']).float()
        test_targets = self._reshape_data(test_targets)

        train_obs = torch.from_numpy(train_windows['obs']).float()
        train_obs = self._reshape_data(train_obs)
        test_obs = torch.from_numpy(test_windows['obs']).float()
        test_obs = self._reshape_data(test_obs)

        return train_obs, train_targets, test_obs, test_targets
    
    def _get_data_set():
        ### define in the child class depending on the dataset
        raise NotImplementedError

    def _wandb_init(self):
        ## Convert Omega Config to Wandb Config (letting wandb know of the config for current run)
        config_dict = OmegaConf.to_container(self.model_cfg, resolve=True, throw_on_missing=True) ###TODO: check if model / global config ?
        expName = self.model_cfg.wandb.exp_name + self.learn_cfg.name
        if self.model_cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"
        ## Initializing wandb object and sweep object
        if self.model_cfg.wandb.log:
            wandb.login(key="55cdb950375b3a8f9ca3d6846e7b2f90b62547f8", relogin=True)
        wandb_run = wandb.init(config=config_dict, project=self.model_cfg.wandb.project_name, name=expName,
                                    mode=mode)  # wandb object has a set of configs associated with it as well 
        return wandb_run
        

    def _train_timeseries_model(self, train_obs, train_targets, test_obs, test_targets):
        """
        Train a timeseries model
        train_obs: (batch_size, seq_len, obs_dim)
        train_targets: (batch_size, seq_len, obs_dim)
        test_obs: (batch_size, seq_len, obs_dim)
        test_targets: (batch_size, seq_len, obs_dim)
        """
        ##### Define WandB Stuffs
        wandb_run = self._wandb_init()

        ### Setting save_path for the model based on the wandb_run id
        if self.learn_cfg.model.load == False:
            save_path = get_original_cwd() + '/experiments/saved_models/' + wandb_run.id + '.ckpt'
        else:
            save_path = get_original_cwd() + '/experiments/saved_models/' + self.learn_cfg.model.id + '.ckpt'

        ### Model Initialize, Train and Inference Modules

        mts3_model = MTS3Simple(input_shape=[train_obs.shape[-1]], config=self.model_cfg)
        ###print the trainable parameters names
        print("Trainable Parameters:..........................")
        for name, param in mts3_model.named_parameters():
            #if param.requires_grad:
            print(name)
            
        mts3_learn = Learn(mts3_model, config=self.model_cfg, run=wandb_run, log=self.model_cfg.wandb['log'])
        if self.model_cfg.learn.data_parallel.enable:
            device_ids = self.model_cfg.learn.data_parallel.device_ids
            print("Device ids are:", device_ids)
            mts3_model = DataParallel(mts3_model, device_ids=device_ids)
            print("Using Data Parallel Model")

        
        if self.model_cfg.learn.model.load == False:
            #### Train the Model (no actions)
            mts3_learn.train(train_obs, train_targets, train_targets, test_obs,
                            test_targets, test_targets)
            
        return mts3_model, wandb_run, save_path
            

    def _test_timeseries_model(self, test_obs, test_targets, normalizer, mts3_model, wandb_run, save_path):
        ##### Inference Module
        dp_infer = Infer(mts3_model, normalizer=normalizer, config=self.model_cfg, run=wandb_run,
                                            log=self.model_cfg.wandb['log'])

        ##### Load best model
        mts3_model.load_state_dict(torch.load(save_path))
        print('>>>>>>>>>>Loaded The Model From Local Folder<<<<<<<<<<<<<<<<<<<')
        ##### Inference From Loaded Model for imputation
        ### Perform inference with no actions
        pred_mean, pred_var, gt, obs_valid, cur_obs, l_prior, l_post = dp_infer.predict(test_obs,
                                                                                test_targets, batch_size=1000, tar=self._data_cfg.tar_type)

        #plotImputation(gt, obs_valid, pred_mean, pred_var, wandb_run, l_prior, l_post, task_labels,  exp_name=namexp)

        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, normalizer,
                                                                tar="observations", denorma=True)
        wandb_run.summary['rmse_denorma_next_state'] = rmse_next_state

        ### Calculate the RMSE for imputation normalized
        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, normalizer,
                                                                tar="observations", denorma=False)
        wandb_run.summary['nrmse_next_state'] = rmse_next_state

        joint_rmse_next_state = joint_rmse(pred_mean, gt, normalizer,
                                            tar="observations", denorma=False)
        for joint in range(joint_rmse_next_state.shape[-1]):
            wandb_run.summary['nrmse_next_state' + "_joint_" + str(joint)] = joint_rmse_next_state[joint]

        print("Root mean square Error is:", rmse_next_state)

        ### Multi Step Inference From Loaded Model
        ### TODO: Create a lot more test sequences

        for step in range(0, test_obs.shape[1]-1):
            if step in [0, 1, int(test_obs.shape[-1] / 2), test_obs.shape[-1] - 1] or step % 300 == 0:
                pred_mean, pred_var, gt, obs_valid, cur_obs, l_prior, l_post = dp_infer.predict_multistep(test_obs,
                                                                                                    test_targets,
                                                                                                    multistep=step,
                                                                                                    batch_size=1000,
                                                                                                    tar=self._data_cfg.tar_type)


                ### Denormalize the predictions and ground truth
                pred_mean_denorm = denorm(pred_mean, normalizer, tar_type=self._data_cfg.tar_type); pred_var_denorm = denorm_var(pred_var, normalizer, 
                                                                                                                                        tar_type=self._data_cfg.tar_type); 
                gt_denorm = denorm(gt, normalizer, tar_type=self._data_cfg.tar_type)
                ### Plot the normalized predictions
                namexp = self.model_cfg.wandb.project_name + "norm_plots/" + str(step) + "/" + self.model_cfg.wandb.exp_name
                plotImputation(gt, obs_valid, pred_mean, pred_var, wandb_run, l_prior, l_post, None, exp_name=namexp)

                #######:::::::::::::::::::Calculate the RMSE and NLL for multistep normalized:::::::::::::::::::::::::::::::::::::
                pred_mean_multistep = pred_mean[:, -self._data_cfg.episode_length:, :]
                pred_var_multistep = pred_var[:, -self._data_cfg.episode_length:, :]
                gt_multistep = gt[:, -self._data_cfg.episode_length:, :]
                rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                        normalizer, tar="observations", denorma=False)
                nll_next_state, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep,
                                                        normalizer, tar="observations", denorma=False)
                wandb_run.summary['norm_nll_multi_step_' + str(step)] = nll_next_state
                wandb_run.summary['nrmse_multistep' + str(step)] = rmse_next_state

                print("Multi Step NRMSE - Step (x.3s) -" + str(step), rmse_next_state)

                #########:::::::::::::::::::Calculate denoramalized RMSE and NLL for multi step ahead predictions:::::::::::::::::::
                rmse_next_state, _, _ = root_mean_squared(pred_mean_multistep, gt_multistep,
                                                                        normalizer,
                                                                        tar="observations", denorma=True)
                nll_next_state, _, _, _ = gaussian_nll(pred_mean_multistep, pred_var_multistep, gt_multistep, normalizer, tar="observations",
                                        denorma=True)
                namexp = self.model_cfg.wandb.project_name + "true_plots/" + str(step) + "/" + self.model_cfg.wandb.exp_name
                plotImputation(gt_denorm, obs_valid, pred_mean_denorm, pred_var_denorm, wandb_run, l_prior, l_post, None, exp_name=namexp)
                wandb_run.summary['rmse_multi_step_' + str(step)] = rmse_next_state
                wandb_run.summary['nll_multi_step_' + str(step)] = nll_next_state

                ## Logging joint wise denormalized multi step ahead predictions
                joint_rmse_next_state = joint_rmse(pred_mean, gt, normalizer,
                                                    tar="observations", denorma=True)
                for joint in range(joint_rmse_next_state.shape[-1]):
                    wandb_run.summary['rmse_multistep_' + str(step) + "_joint_" + str(joint)] = joint_rmse_next_state[
                        joint]









def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()