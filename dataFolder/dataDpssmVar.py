import sys
import os

sys.path.append('.')
from torch.utils.data import Dataset
import numpy as np
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pandas as pd
from utils.dataProcess import normalize, denormalize


class metaData(Dataset):
    def __init__(self, data_cfg=None):
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg
            self.c = data_cfg

        self._long_term_pred = self.c.long_term_pred

        self._save_windows = self.c.save
        self._load_windows = self.c.load
        self.trajPerTask = self.c.trajPerTask
        self._down_sample = self.c.downsample

        self.tar_type = self.c.tar_type
        self._obs_impu = self.c.obs_imp
        self._task_impu = self.c.task_imp

        self._shuffle_split = self.c.shuffle_split

        self.episode_length = self.c.episode_length  # window length for a single hiprssm instance
        self.num_episodes = self.c.num_episodes # number of hiprssm instances
        self.normalizer = None
        self.filename = self.c.file_name
        self.standardize = self.c.standardize
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}



    def get_statistics(self, data, dim, difference=False):
        if difference:
            data = (data[:, 1:, :dim] - data[:, :-1, :dim])
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1] * x.shape[2], -1))
        data = reshape(data);
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def _load_trajectories(self):
        """
                Load from disk the obs, act, next_obs, task_labels
                """
        return NotImplementedError

    def _pre_process(self, train_obs, train_act, train_next_obs, train_tasks, test_obs=None, test_act=None, test_next_obs=None, test_tasks=None):
        print(".........Before Preprocessing...........", train_obs.shape, train_act.shape, train_next_obs.shape, train_tasks.shape)
        if test_obs is None:
            # train test split
            train_obs, train_act, train_next_obs, train_tasks, test_obs, test_act, test_next_obs, test_tasks = self.train_test_split(
                train_obs, train_act,
                train_next_obs, train_tasks)



        train_delta = train_next_obs - train_obs
        test_delta = test_next_obs - test_obs

        # get different statistics for state, actions, delta_state, delta_action and residuals which will be used for standardization
        mean_state_diff, std_state_diff = self.get_statistics(train_delta, dim=train_delta.shape[-1], difference=False)
        mean_obs, std_obs = self.get_statistics(train_obs, dim=train_obs.shape[-1])
        mean_act, std_act = self.get_statistics(train_act, dim=train_act.shape[-1])
        self.normalizer = dict()
        self.normalizer['observations'] = [mean_obs, std_obs]
        self.normalizer['actions'] = [mean_act, std_act]
        self.normalizer['diff'] = [mean_state_diff, std_state_diff]

        # compute delta
        if self.tar_type == 'delta':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Differences <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            self.normalizer['targets'] = [mean_state_diff, std_state_diff]
        else:
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Next States(not differences) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            self.normalizer['targets'] = [mean_obs, std_obs]

        # Standardize
        if self.standardize:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>Standardizing The Data<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            train_obs = normalize(train_obs, self.normalizer["observations"][0],
                                       self.normalizer["observations"][1])
            train_acts = normalize(train_act, self.normalizer["actions"][0],
                                        self.normalizer["actions"][1])

            test_obs = normalize(test_obs, self.normalizer["observations"][0],
                                      self.normalizer["observations"][1])
            test_acts = normalize(test_act, self.normalizer["actions"][0],
                                       self.normalizer["actions"][1])

            if self.tar_type == 'delta':
                train_targets = normalize(train_delta, self.normalizer["diff"][0],
                                               self.normalizer["diff"][1])
                test_targets = normalize(test_delta, self.normalizer["diff"][0],
                                              self.normalizer["diff"][1])
            else:
                train_targets = normalize(train_next_obs, self.normalizer["observations"][0],
                                               self.normalizer["observations"][1])
                test_targets = normalize(test_next_obs, self.normalizer["observations"][0],
                                              self.normalizer["observations"][1])


        else:
            train_obs = train_obs
            train_acts = train_act
            test_obs = test_obs
            test_acts = test_act
            if self.tar_type == 'delta':
                train_targets = train_delta
                test_targets = test_delta
            else:
                train_targets = train_next_obs
                test_targets = test_next_obs
        train_task_idx = train_tasks
        test_task_idx = test_tasks

        train_obs_valid_batch, train_task_valid_batch = self._create_valid_flags(train_obs)
        test_obs_valid_batch, test_task_valid_batch = self._create_valid_flags(test_obs)

        train_data_windows = {'obs': train_obs, 'act': train_acts, 'target': train_targets, 'obs_valid': train_obs_valid_batch, "task_valid": train_task_valid_batch,
                        'normalization': self.normalizer,
                        'task_index': np.mean(train_task_idx,-1)}  ###CLEVER TRICK %trajPerTask
        test_data_windows = {'obs': test_obs, 'act': test_acts, 'target': test_targets,
                              'obs_valid': test_obs_valid_batch, "task_valid": test_task_valid_batch,
                              'normalization': self.normalizer,
                              'task_index': np.mean(test_task_idx ,-1)}  ###CLEVER TRICK %trajPerTask
        # TODO for the target(second half) initialize few things to True

        return train_data_windows, test_data_windows

    def _create_valid_flags(self,obs):
        rs = np.random.RandomState(seed=42)
        obs_valid_batch = rs.rand(obs.shape[0], obs.shape[1], obs.shape[2], 1) < 1 - self._obs_impu
        if self.c.episodic:
            obs_valid_batch[:, :, :5] = True
        else:
            obs_valid_batch[:, 0, :5] = True

        if self._long_term_pred > 1:
            # set obs_valid for meta-step to False when task_valid is false (to verify)
            # TODO Verify
            task_valid_batch = rs.rand(obs.shape[0], obs.shape[1], 1) < 1 - self._task_impu
            task_valid_batch[:, -self._long_term_pred:] = False
            task_valid_batch[:, :1] = True
            # obs_valid_batch[:, -self._long_term_pred:, :] = False
        else:
            task_valid_batch = rs.rand(obs.shape[0], obs.shape[1], 1) < 1 - self._task_impu
            task_valid_batch[:, :1] = True

        return np.array(obs_valid_batch), np.array(task_valid_batch)



if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

