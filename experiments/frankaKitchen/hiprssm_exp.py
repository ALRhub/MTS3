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

from dataFolder.fkDataDpssm import metaFkData
from experiments.exp_prediction_hiprssm import Experiment
from agent.worldModels import MTS3
from hydra.utils import get_original_cwd, to_absolute_path


nn = torch.nn

@hydra.main(config_path='conf',config_name="config")
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg
    exp = Experiment(model_cfg)

    train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer = exp._get_data_set()
    ### train the model
    mts3_model, wandb_run, save_path = exp._train_world_model(train_obs, train_act, train_targets, test_obs, test_act, test_targets)
    ### test the model
    #TODO: normalizer format specify
    exp._test_world_model(test_obs, test_act, test_targets, normalizer, mts3_model, wandb_run, save_path)


class Experiment(Experiment):
    def __init__(self, cfg):
        super(Experiment, self).__init__(cfg)

    def _load_save_train_test_data(self, dataLoaderClass):
        """
        write a function to load the data and return the train and test data
        :return: train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer
        """
        with open(get_original_cwd() + self._data_train_cfg.save_path, 'rb') as f:
            data_dict = pickle.load(f)
            print("Train Obs Shape", data_dict['train_obs'].shape)
            print("Train Act Shape", data_dict['train_act'].shape)
            print("Train Targets Shape", data_dict['train_targets'].shape)
            print("Test Obs Shape", data_dict['test_obs'].shape)
            print("Test Act Shape", data_dict['test_act'].shape)
            print("Test Targets Shape", data_dict['test_targets'].shape)
            print("Normalizer", data_dict['normalizer'])
        return data_dict['train_obs'], data_dict['train_act'], data_dict['train_targets'], data_dict['test_obs'], \
        data_dict['test_act'], data_dict['test_targets'], data_dict['normalizer']

    def _get_data_set(self):
        """
        write a function to load the data and return the train and test data (this is where all the data processing
        happens and can be userdefined)
        :return: train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer
        """
        tar_type = self._data_train_cfg.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states
        assert self._data_train_cfg.tar_type == self._data_test_cfg.tar_type  # "Train and Test Target Types are same"

        ### load or generate data
        train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer = self._load_save_train_test_data(
            metaFkData)

        ### Convert data to tensor

        ## choose first 100 time steps
        # train_obs = train_obs[:, :200, :]
        # train_act = train_act[:, :200, :]
        # train_targets = train_targets[:, :200, :]
        # test_obs = test_obs[:, :200, :]
        # test_act = test_act[:, :200, :]
        # test_targets = test_targets[:, :200, :]

        return train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()