import os
import time as t
from typing import Tuple
import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from agent.worldModels.MTS3Simple import MTS3Simple
from utils.dataProcess import diffToState, diffToStateImpute

optim = torch.optim
nn = torch.nn


class Infer:

    def __init__(self, model: MTS3Simple, normalizer, config = None, run = None, log=True, use_cuda_if_available: bool = True):
        """
        Inferece class for "Unactuated" MTS3 model (MTS3Simple)
        :param model: model to evaluate
        :param normalizer: normalizer used to normalize the data
        :param config: config dict
        :param run: wandb run
        :param log: whether to log to wandb
        :param use_cuda_if_available: whether to use cuda if available
        """
        assert run is not None, 'pass a valid wandb run'
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._normalizer = normalizer
        self._model = model
        self._obs_imp = 0.5
        self._task_impu = 0
        self._exp_name = run.name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = bool(log)

        if self._log:
            self._run = run


    def _create_valid_flags_dream(self,num_samples,burn_in,steps):
        ##FIXME: this is just a logic , actual dreaming may need access to hidden states
        """
        When you dream, from current time step / state imagine next few steps"""
        seed = np.random.randint(1, 1000)
        rs = np.random.RandomState(seed=seed)

        obs_valid_batch = rs.rand(num_samples, burn_in+steps, 1) < 1 #every obs is valid
        task_valid_batch = rs.rand(num_samples, burn_in+steps, 1) < 1 #every task is valid

        ## ceil of steps/H gives number of task steps
        task_steps = int(np.ceil(steps/self.c.mts3.time_scale_multiplier))

        ## convert to torch
        obs_valid_batch = torch.from_numpy(obs_valid_batch).bool()
        task_valid_batch = torch.from_numpy(task_valid_batch).bool()
        return  obs_valid_batch, task_valid_batch
    
    def _create_valid_flags(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create valid flags for observations and tasks for imputation
        :param obs: observations
        :return: obs_valid, task_valid
        """
        seed = np.random.randint(1, 1000)
        rs = np.random.RandomState(seed=seed)
        num_managers = int(np.ceil(obs.shape[1] / self.c.mts3.time_scale_multiplier))

        obs_valid = rs.rand(obs.shape[0], obs.shape[1], 1) < self._obs_imp
        task_valid = rs.rand(obs.shape[0], num_managers, 1) < self._task_impu

        obs_valid = torch.from_numpy(obs_valid).bool()
        task_valid = torch.from_numpy(task_valid).bool()
        return obs_valid, task_valid
    
    def _create_valid_flags_multistep(self,obs,steps):
        """
        Create valid flags with last "steps/task_steps" to False
        :param steps: number of steps to be predicted (worker)
        """
        seed = np.random.randint(1, 1000)
        rs = np.random.RandomState(seed=seed)
        num_managers = int(np.ceil(obs.shape[1] / self.c.mts3.time_scale_multiplier))

        obs_valid_batch = rs.rand(obs.shape[0], obs.shape[1], 1) < 1 #every obs is valid
        task_valid_batch = rs.rand(obs.shape[0], num_managers, 1) < 1 #every task is valid

        ## ceil of steps/H gives number of task steps
        task_steps = int(np.ceil(steps/self.c.mts3.time_scale_multiplier))

        ## set obs_valid to False for last steps   
        obs_valid_batch[:, -steps:, :] = False
        task_valid_batch[:, -task_steps:, :] = False

        ## convert to torch
        obs_valid_batch = torch.from_numpy(obs_valid_batch).bool()
        task_valid_batch = torch.from_numpy(task_valid_batch).bool()
        return  obs_valid_batch, task_valid_batch



    def predict(self, obs: np.ndarray, targets: np.ndarray, batch_size: int = -1,  tar="observations") -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param targets: targets to evaluate on
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
            data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        obs_valid, task_valid = self._create_valid_flags(obs)
        # print(train_obs_valid.shape, train_task_valid.shape)
        dataset = TensorDataset(obs, targets, obs_valid, task_valid)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        cur_obs_list = []
        l_prior_vis_list = []
        l_post_vis_list = []
        task_id_list = []
        gt_list = []
        out_mean_list = []
        out_var_list = []
        obs_valid_list = []
        task_valid_list = []

        for batch_idx, (obs_batch, targets_batch, obs_valid_batch, task_valid) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = (obs_batch).to(self._device)
                target_batch = (targets_batch).to(self._device)
                obs_valid_batch = (obs_valid_batch).to(self._device)
                task_valid_batch = (task_valid).to(self._device)

                ##TODO: How to get obs_valid, task_valid etc ?? 

                # Forward Pass
                out_mean, out_var, mu_l_prior, cov_l_prior, mu_l_post, cov_l_post = self._model(obs_batch, obs_valid_batch,task_valid_batch)

                # Diff To State
                if tar == "delta":
                    # when predicting differences convert back to actual observations (common procedure in model based RL and dynamics learning)
                    out_mean = \
                        torch.from_numpy(
                            diffToStateImpute(out_mean, obs_batch, masked_obs_valid_batch, self._normalizer,
                                                standardize=True)[0]) ## TODO: Recheck is obs_valid or masked_obs_valid_batch
                    target_batch = \
                        torch.from_numpy(
                            diffToState(target_batch, obs_batch, self._normalizer, standardize=True)[0])

                out_mean_list.append(out_mean.cpu().numpy())
                out_var_list.append(out_var.cpu().numpy())
                gt_list.append(target_batch.cpu().numpy())  # if test_gt_known flag is False then we get list of Nones
                obs_valid_list.append(obs_valid_batch.cpu().numpy())
                task_valid_list.append(task_valid_batch.cpu().numpy())
                cur_obs_list.append(obs_batch.cpu().numpy())
                l_prior_vis_list.append(mu_l_prior.detach().cpu().numpy())
                l_post_vis_list.append(mu_l_post.detach().cpu().numpy())

        out_mean = np.concatenate(out_mean_list, axis=0)
        out_var = np.concatenate(out_var_list, axis=0)
        gt_obs = np.concatenate(gt_list, axis=0)
        current_obs = np.concatenate(cur_obs_list, axis=0)
        obs_valid = np.concatenate(obs_valid_list, axis=0)
        l_vis_prior = np.concatenate(l_prior_vis_list, axis=0)
        l_vis_post = np.concatenate(l_post_vis_list, axis=0)
        return out_mean, out_var, gt_obs, obs_valid, current_obs, l_vis_prior, l_vis_post

    def predict_multistep(self, obs: np.ndarray, targets: np.ndarray, multistep=1,
                batch_size: int = -1, tar="observations") -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param targets: targets to evaluate on
        :param multistep: how many task level multistep predictions to be done
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
            data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        obs_valid, task_valid = self._create_valid_flags_multistep(obs, multistep)

        dataset = TensorDataset(obs, targets, obs_valid, task_valid)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        cur_obs_list = []
        l_prior_vis_list = []
        l_post_vis_list = []
        gt_list = []
        out_mean_list = []
        out_var_list = []
        obs_valid_list = []
        task_valid_list = []

        for batch_idx, (obs_batch, targets_batch, obs_valid_batch, task_valid) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                target_batch = (targets_batch).to(self._device)
                obs_valid_batch = (obs_valid_batch).to(self._device)
                task_valid_batch = (task_valid).to(self._device)

                # Forward Pass
                out_mean, out_var, mu_l_prior, cov_l_prior, mu_l_post, cov_l_post = self._model(obs_batch, obs_valid_batch,task_valid_batch)

                # Diff To State
                if tar == "delta":
                    # when predicting differences convert back to actual observations (common procedure in model based RL and dynamics learning)
                    out_mean = \
                        torch.from_numpy(
                            diffToStateImpute(out_mean, obs_batch, obs_valid, self._normalizer,
                                                standardize=True)[0]) ##TODO: Recheck is obs_valid or masked_obs_valid_batch
                    target_batch = \
                        torch.from_numpy(
                            diffToState(target_batch, obs_batch, self._normalizer, standardize=True)[0])

                out_mean_list.append(out_mean.cpu().numpy())
                out_var_list.append(out_var.cpu().numpy())
                gt_list.append(target_batch.cpu().numpy())  # if test_gt_known flag is False then we get list of Nones
                obs_valid_list.append(obs_valid_batch.cpu().numpy())
                task_valid_list.append(task_valid_batch.cpu().numpy())
                cur_obs_list.append(obs_batch.cpu().numpy())
                l_prior_vis_list.append(mu_l_prior.detach().cpu().numpy())
                l_post_vis_list.append(mu_l_post.detach().cpu().numpy())

        out_mean = np.concatenate(out_mean_list, axis=0)
        out_var = np.concatenate(out_var_list, axis=0)
        gt_obs = np.concatenate(gt_list, axis=0)
        obs_valid = np.concatenate(obs_valid_list, axis=0)
        current_obs = np.concatenate(cur_obs_list, axis=0)
        l_vis_prior = np.concatenate(l_prior_vis_list, axis=0)
        l_vis_post = np.concatenate(l_post_vis_list, axis=0)
        return out_mean, out_var, gt_obs, obs_valid, current_obs, l_vis_prior, l_vis_post

    