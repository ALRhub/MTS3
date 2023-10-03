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

from dataFolder.excavatorDataVar import metaExData
from experiments.exp_prediction_mts3 import Experiment
from agent.worldModels import MTS3


nn = torch.nn

@hydra.main(config_path='conf',config_name="config")
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg
    exp = MobileExperiment(model_cfg)

    train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer = exp._get_data_set()
    ### train the model
    mts3_model, wandb_run, save_path = exp._train_world_model(train_obs, train_act, train_targets, test_obs, test_act, test_targets)
    ### test the model
    #TODO: normalizer format specify
    exp._test_world_model(test_obs, test_act, test_targets, normalizer, mts3_model, wandb_run, save_path)


class MobileExperiment(Experiment):
    def __init__(self, cfg):
        super(MobileExperiment, self).__init__(cfg)

    def _get_data_set(self):
        """
        Write your own function to load the data and return the train and test data (observations, actions and next_state).
        Shape of the train and test data should be (num_samples, time_steps, num_features)
        Normalizer TODO: specify the format
        :return: train_obs, train_act, train_targets, test_obs, test_act, test_targets, normalizer
        """
        ####### Write your own code here ########
        assert self._data_train_cfg.tar_type == self._data_test_cfg.tar_type #"Train and Test Target Types are same"

        ### load or generate data
        data, data_test = self._load_save_train_test_data(metaExData)

        ### Convert data to tensor
        train_obs, train_act, train_targets, _, _, _ = self._convert_to_tensor_reshape(data)
        _, _, _, test_obs, test_act, test_targets = self._convert_to_tensor_reshape(data_test)

        #########################################
        return train_obs, train_act, train_targets, test_obs, test_act, test_targets, data.normalizer


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()