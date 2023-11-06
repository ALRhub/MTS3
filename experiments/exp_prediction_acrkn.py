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
from torchview import draw_graph

from agent.worldModels.acRKN import acRKN
from agent.Learn.repre_learn_rnn import Learn
from agent.Infer.repre_infer_rnn import Infer
from utils.metrics import naive_baseline
from utils.dataProcess import split_k_m, denorm, denorm_var
from utils.metrics import root_mean_squared, joint_rmse, gaussian_nll
from hydra.utils import get_original_cwd, to_absolute_path
from utils.plotTrajectory import plotImputation, plotMbrl, plotLongTerm

nn = torch.nn


class Experiment():
    """
    Experiment class for training and testing the world model (Actuated MTS3 Model)"""

    def __init__(self, cfg):
        self.model_cfg = cfg.model
        self.learn_cfg = self.model_cfg.learn
        self._data_cfg = self.model_cfg.data
        # 'next_state' - if to trian directly on the  next states
        torch.cuda.empty_cache()

    def _reshape_data(self, data):
        ## reshape the data by flattening the second and third dimension
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], -1)
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

        train_act = torch.from_numpy(train_windows['act']).float()
        train_act = self._reshape_data(train_act)
        test_act = torch.from_numpy(test_windows['act']).float()
        test_act = self._reshape_data(test_act)

        return train_obs, train_act, train_targets, test_obs, test_act, test_targets

    def _get_data_set():
        ### define in the child class depending on the dataset
        raise NotImplementedError

    def _wandb_init(self):
        ## Convert Omega Config to Wandb Config (letting wandb know of the config for current run)
        config_dict = OmegaConf.to_container(self.model_cfg, resolve=True,
                                                throw_on_missing=True)  ###TODO: check if model / global config ?
        expName = self.model_cfg.wandb.exp_name + self.learn_cfg.name
        if self.model_cfg.wandb.log:
            mode = 