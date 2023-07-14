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
from experiments.exp_prediction_mts3 import Experiment
from agent.worldModels import MTS3


nn = torch.nn

@hydra.main(config_path='conf_dp',config_name="config")
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg
    exp = Experiment(model_cfg)

    train_obs, train_act, train_targets, test_obs, test_act, test_targets = exp._get_data_set()
    ### train the model
    exp._train_world_model(train_obs, train_act, train_targets, test_obs, test_act, test_targets)
    ### test the model
    #exp._test_world_model(test_obs, test_act, test_targets)


class MobileExperiment(Experiment):
    def __init__(self, cfg):
        super(Experiment, self).__init__(cfg)

    def _get_data_set():
        tar_type = self._data_train_cfg.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states
        assert self._data_train_cfg.tar_type == self._data_test_cfg.tar_type #"Train and Test Target Types are same"

        ### load or generate data
        data, data_test = self._load_save_data(metaMobileData)

        ### Convert data to tensor
        train_obs, train_act, train_targets, _, _, _ = self._convert_to_tensor(data)
        _, _, _, test_obs, test_act, test_targets = self._convert_to_tensor(data_test)

        return train_obs, train_act, train_targets, test_obs, test_act, test_targets


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()