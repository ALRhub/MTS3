import sys
import os

sys.path.append('.')
from dataFolder.dataDpssm import metaData
import numpy as np
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import pandas as pd
from utils.dataProcess import normalize, denormalize


class metaMazeData(metaData):
    def __init__(self, data_cfg=None):
        super(metaMazeData, self).__init__(data_cfg)
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg

        if self.c.type == 'm_l':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = get_original_cwd() + '/dataFolder/d4rl/maze2d-large-v1.pkl'
        elif self.c.type == 'm_l5':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = get_original_cwd() + '/dataFolder/d4rl/maze2d-large-v1-50k.pkl'
        elif self.c.type == 'u':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = get_original_cwd() + '/dataFolder/d4rl/maze2d-udense-v1.pkl'
        elif self.c.type == 'medium':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = get_original_cwd() + '/dataFolder/d4rl/maze2d-medium-v1.pkl'
        elif self.c.type == 'u-ant':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = get_original_cwd() + '/dataFolder/d4rl/antmaze-umaze-diverse-v0-1k.pkl'
        elif self.c.type == 'andro':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = get_original_cwd() + '/dataFolder/d4rl/hammer-cloned-v0-5k.pkl'
        else:
            ### Raise python Error enter valid type
            raise Exception('Please specify a valid type for data')

        # self._trajectoryPath = '/home/vshaj/CLAS/meta-ssm/dataFolder/MujocoFolder/dyn_model.pickle'

        self.trajPerTask = self.c.trajPerTask

        self.tar_type = self.c.tar_type

        self._split = OmegaConf.to_container(self.c.split)
        self._shuffle_split = self.c.shuffle_split

        self.num_training_sequences = self.c.num_training_sequences
        self.num_testing_sequences = self.c.num_testing_sequences
        self.episode_length = self.c.episode_length  # window length for a single hiprssm instance
        self.num_episodes = self.c.num_episodes  # number of hiprssm instances
        self.normalizer = None
        self.standardize = self.c.standardize
        obs, act, next_obs, tasks, rewards = self._load_trajectories()
        print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,)
        self.train_windows, self.test_windows = self._pre_process(obs, act, next_obs, tasks)
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}

    def _load_trajectories(self):
        # collect obs, act, next states
        if self.c.type == 'u-ant':
            dim = 13
        elif self.c.type == 'andro':
            dim = 100
        else:
            dim = 2
        data = {'observations': [], 'actions': [], 'next_observations': [], 'rewards': []}
        if self._trajectoryPath is None:
            n = 0
            for filename in os.listdir(self._dataFolder):
                if filename.endswith(".pickle"):
                    self._trajectoryPath = self._dataFolder + filename
                    with open(self._trajectoryPath, 'rb') as f:
                        data_np = pickle.load(f)
                        if n == 0:
                            data['observations'] = np.array(data_np['observations'][:, :-1,:dim])
                            data['actions'] = np.array(data_np['actions'][:, :-1])
                            # data['tasks'] = np.array(data_np['tasks'][:,:-1,:])
                            data['next_observations'] = np.array(data_np['observations'][:, 1:,:dim])
                            data['rewards'] = np.array(data_np['rewards'][:, :-1])
                            n = n + 1
                            continue
                        else:
                            ## concatenate the data
                            print(data_np['observations'].shape, data['observations'].shape)
                            data['observations'] = np.concatenate(
                                (data['observations'], data_np['observations'][:, :-1, :dim]), axis=0)
                            data['actions'] = np.concatenate((data['actions'], data_np['actions'][:, :-1]), axis=0)
                            # data['tasks'] = np.concatenate((data['tasks'], data_np['tasks'][:,:-1,:]), axis=0)
                            data['next_observations'] = np.concatenate(
                                (data['next_observations'], data_np['observations'][:, 1:, :dim]), axis=0)
                            data['rewards'] = np.concatenate((data['rewards'], data_np['rewards'][:, :-1]), axis=0)
        else:
            with open(self._trajectoryPath, 'rb') as f:
                data_np = pickle.load(f)
            print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np['observations'].shape)
            # dim = np.r_[0:15, 30:41]
            data['observations'] = data_np['observations'][:, :-1,:dim]
            data['actions'] = data_np['actions'][:, :-1]
            # data['tasks'] = data_np['tasks'][:,:-1,:]
            data['next_observations'] = data_np['observations'][:, 1:, :dim]
            data['rewards'] = data_np['rewards'][:, :-1]
        obs = data['observations']
        act = data['actions']
        print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,
              act.shape)
        next_obs = data['next_observations']
        tasks = data['rewards']
        rewards = data['rewards']
        return obs, act, next_obs, tasks, rewards


if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

