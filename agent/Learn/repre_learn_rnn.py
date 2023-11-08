##CHANGED: valid flag logic
##CHANGED: what model gets and throws out
##CHANGED: latent_vis separate new module
##CHANGED: curiculum learning strategy defined properly

import os
import time as t
from typing import Tuple
import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from hydra.utils import get_original_cwd, to_absolute_path

from agent.worldModels.hipRSSM import hipRSSM
from utils.Losses import mse, gaussian_nll
from utils.PositionEmbedding import PositionEmbedding as pe
from utils.plotTrajectory import plotImputation

optim = torch.optim
nn = torch.nn


class Learn:

    def __init__(self, model: hipRSSM, config = None, run=None, log=True, use_cuda_if_available: bool = True):
        """
        :param model: nn module for np_dynamics
        :param loss: type of loss to train on 'nll' or 'mse'
        :param use_cuda_if_available: if gpu training set to True
        """
        assert run is not None, 'pass a valid wandb run'
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._model = model
        self._pe = pe(self._device)
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config
        self._loss = self.c.learn.loss
        self._obs_impu = self.c.learn.obs_imp
        self._exp_name = run.name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._learning_rate = self.c.learn.lr
        self._save_path = get_original_cwd() + '/experiments/saved_models/' + run.id + '.ckpt'
        self._latent_visualization = self.c.learn.latent_visualization
        self._epochs = self.c.learn.epochs
        self._batch_size = self.c.learn.batch_size

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = bool(log)
        self.save_model = self.c.learn.save_model
        self.vis_dim = self._model._lod
        if self._log:
            self._run = run

    def _create_valid_flags(self, obs, train=False):
        """
        Create valid flags for worker and manager
        :param obs: observations
        :param train: if True create valid flags for training else for testing
        :return: obs_valid, task_valid
        """
        seed = np.random.randint(1, 1000)
        rs = np.random.RandomState(seed=seed)
        if train:
            obs_valid_batch = rs.rand(obs.shape[0], obs.shape[1], 1) < 1 - self._obs_impu

        else:
            obs_valid_batch = rs.rand(obs.shape[0], obs.shape[1], 1) < 1 - self._obs_impu

        return torch.from_numpy(obs_valid_batch).bool()

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray, train_targets: np.ndarray,
                    train_obs_valid: np.ndarray, train_task_idx: np.ndarray,
                    batch_size: int) -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param train_task_idx: task ids per episode
        :param batch_size: batch size for each gradient update
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_targets, train_obs_valid, train_task_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        avg_loss = avg_metric_nll = avg_metric_mse = 0
        t0 = t.time()
        l_prior_vis_list = []
        l_post_vis_list = []
        task_id_list = []
        act_vis_list = []
        for batch_idx, (obs, act, targets, obs_valid, task_id) in enumerate(loader):
            # Assign tensors to device
            obs_batch = (obs).to(self._device)
            act_batch = act.to(self._device)
            target_batch = (targets).to(self._device)
            obs_valid_batch = (obs_valid).to(self._device)
            task_id = (task_id).to(self._device)

            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            # Forward Pass
            out_mean, out_var = self._model(obs_batch, act_batch, obs_valid_batch)  ##TODO: check if train=True is needed

            ## Calculate Loss
            if self._loss == 'nll':
                loss = gaussian_nll(target_batch, out_mean, out_var)
            else:
                loss = mse(target_batch, out_mean)

            ### viz graph
            # print('>>>>>>>>>>>Viz Graph<<<<<<<<<<<<<<')

            # make_dot(loss, params=dict(self._model.named_parameters())).render("attached", format="png")
            # Backward Pass
            loss.backward()  # FIXME: check if this is needed

            # Clip Gradients
            if self.c.learn.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

            with torch.no_grad():  #
                metric_nll = gaussian_nll(target_batch, out_mean, out_var)
                metric_mse = mse(target_batch, out_mean)


            avg_loss += loss.detach().cpu().numpy()
            avg_metric_nll += metric_nll.detach().cpu().numpy()
            avg_metric_mse += metric_mse.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an epoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        with torch.no_grad():
            self._tr_sample_gt = target_batch.detach().cpu().numpy()
            self._tr_sample_valid = obs_valid_batch.detach().cpu().numpy()
            self._tr_sample_pred_mu = out_mean.detach().cpu().numpy()
            self._tr_sample_pred_var = out_var.detach().cpu().numpy()

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))

        return avg_loss, avg_metric_nll, avg_metric_rmse, t.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, targets: np.ndarray, obs_valid: np.ndarray,
                                task_idx: np.ndarray, batch_size: int = -1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param targets: targets to evaluate on
        :param task_idx: task index
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
            data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        dataset = TensorDataset(obs, act, targets, obs_valid, task_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        avg_loss = avg_metric_nll = avg_metric_mse = 0.0
        avg_metric = 0.0

        for batch_idx, (obs_batch, act_batch, targets_batch, obs_valid_batch, task_id) in enumerate(
                loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                target_batch = (targets_batch).to(self._device)
                obs_valid_batch = (obs_valid_batch).to(self._device)

                # Forward Pass
                out_mean, out_var = self._model(obs_batch, act_batch, obs_valid_batch)

                self._te_sample_gt = target_batch.detach().cpu().numpy()
                self._te_sample_valid = obs_valid_batch.detach().cpu().numpy()
                self._te_sample_pred_mu = out_mean.detach().cpu().numpy()
                self._te_sample_pred_var = out_var.detach().cpu().numpy()

                ## Calculate Loss
                if self._loss == 'nll':
                    loss = gaussian_nll(target_batch, out_mean, out_var)
                else:
                    loss = mse(target_batch, out_mean)

                metric_nll = gaussian_nll(target_batch, out_mean, out_var)
                metric_mse = mse(target_batch, out_mean)

                avg_loss += loss.detach().cpu().numpy()
                avg_metric_nll += metric_nll.detach().cpu().numpy()
                avg_metric_mse += metric_mse.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))

        return avg_loss, avg_metric_nll, avg_metric_rmse

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor,
                train_targets: torch.Tensor, train_task_idx: torch.Tensor,
                val_obs: torch.Tensor = None, val_act: torch.Tensor = None,
                val_targets: torch.Tensor = None, val_task_idx: torch.Tensor = None, val_interval: int = 1,
                val_batch_size: int = -1) -> None:
        '''
        :param train_obs: training observations for the model
        :param train_act: training actions for the model
        :param train_targets: training targets for the model
        :param train_task_idx: task_index for different training sequence
        :param epochs: number of epochs to train on
        :param batch_size: batch_size for gradient descent
        :param val_obs: validation observations for the model (includes context and targets)
        :param val_act: validation actions for the model (includes context and targets)
        :param val_targets: validation targets for the model (includes context and targets)
        :param val_task_idx: task_index for different testing sequence
        :param val_interval: how often to perform validation
        :param val_batch_size: batch_size while performing inference
        :return:
        '''

        """ Train Loop"""
        torch.cuda.empty_cache()  #### Empty Cache
        if val_batch_size == -1:
            val_batch_size = 1 * self._batch_size
        best_loss = np.inf
        best_nll = np.inf
        best_rmse = np.inf
        if self.vis_dim:
            assert train_task_idx is not None, 'Pass train_task_idx for latent visualization'
            assert val_task_idx is not None, 'Pass val_task_idx for latent visualization'

        if self._log:
            wandb.watch(self._model, log='all', log_freq=1)
            artifact = wandb.Artifact('saved_model', type='model')
            #wandb.log({"Config": OmegaConf.to_container(self.c)})

        ### Curriculum Learning Strategy
        curriculum_num = 0
        if self._epochs <= 250:
            curriculum_switch = 5
        else:
            curriculum_switch = 10
        ### Loop over epochs and train
        for i in range(self._epochs):
            ### Create Valid Flags
            train_obs_valid = self._create_valid_flags(train_obs, train=True)
            val_obs_valid = self._create_valid_flags(val_obs)

            print("Fraction of Valid Train and Test Task:",
                        np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape),
                        np.count_nonzero(val_obs_valid) / np.prod(val_obs_valid.shape))

            train_loss, train_metric_nll, train_metric_rmse, time = self.train_step(
                train_obs,
                train_act,
                train_targets,
                train_obs_valid,
                train_task_idx,
                self._batch_size)

            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(
                i + 1, self._loss, train_loss, 'target_nll:', train_metric_nll, 'target_rmse:', train_metric_rmse,
                time))
            if np.any(
                    np.isnan(train_loss)):
                print("-------------------------NaN Encountered------------------------")
            assert not np.any(
                np.isnan(train_loss)), "Result contained NaN: {train_loss}"  # sanity check for NaNs in loss
            # self._writer.add_scalar(self._loss + "/train_loss", train_loss, i)
            # self._writer.add_scalar("nll/train_metric", train_metric_nll, i)
            # self._writer.add_scalar("rmse/train_metric", train_metric_rmse, i)
            if self._log:
                wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll,
                            "rmse/train_metric": train_metric_rmse, "epochs": i})

            ###
            # wandb.watch(self._model, log='all', log_freq=20)

            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_loss, val_metric_nll, val_metric_rmse = self.eval(
                    val_obs,
                    val_act,
                    val_targets,
                    val_obs_valid,
                    val_task_idx,
                    val_batch_size)
                if val_loss < best_loss:
                    if self.save_model:
                        print('>>>>>>>Saving Best Model<<<<<<<<<<')
                        torch.save(self._model.state_dict(), self._save_path)
                    if self._log:
                        wandb.run.summary['best_loss'] = val_loss
                    best_loss = val_loss
                if val_metric_nll < best_nll:
                    if self._log:
                        wandb.run.summary['best_nll'] = val_metric_nll
                    best_nll = val_metric_nll
                if val_metric_rmse < best_rmse:
                    if self._log:
                        wandb.run.summary['best_rmse'] = val_metric_rmse
                    best_rmse = val_metric_rmse
                print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll',
                                                                                val_metric_nll, 'target_rmse',
                                                                                val_metric_rmse))

                if self._log:
                    wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll,
                                "rmse/test_metric": val_metric_rmse, "epochs": i})


        if self.c.learn.save_model:
            artifact.add_file(self._save_path)
            wandb.log_artifact(artifact)

        if self.c.learn.plot_traj:
            plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,
                            self._run, log_name='train', exp_name=self._exp_name)
            plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,
                            self._run, log_name='test', exp_name=self._exp_name)

