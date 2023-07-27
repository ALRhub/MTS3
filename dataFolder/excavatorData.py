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


class metaExData(metaData):
    def __init__(self, data_cfg=None):
        super(metaExData, self).__init__(data_cfg)
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg


        self._trainTrajectoryPath = '/home/vshaj/CLAS/DP-SSM-v2/dataFolder/actuator_model_v2/data/export/train.pickle'
        self._valTrajectoryPath = '/home/vshaj/CLAS/DP-SSM-v2/dataFolder/actuator_model_v2/data/export/val.pickle'
        self._testTrajectoryPath = '/home/vshaj/CLAS/DP-SSM-v2/dataFolder/actuator_model_v2/data/export/test.pickle'


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
        obs, act, next_obs, tasks = self._load_trajectories(type='train')
        val_obs, val_act, val_next_obs, val_tasks = self._load_trajectories(type='val')
        test_obs, test_act, test_next_obs, test_tasks = self._load_trajectories(type='test')
        self.train_windows, self.val_windows = self._pre_process(obs, act, next_obs, tasks, None, val_obs, val_act, val_next_obs, val_tasks)
        self.test_windows = self._pre_process(test_obs, test_act, test_next_obs, test_tasks)
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}

    def _load_trajectories(self,type='train'):
        # collect obs, act, next states
        data = {'observations': [], 'actions': [], 'next_observations': [], 'rewards': []}
        n=0
        ## load from self._trajectoryPath
        if type=='train':
            with open(self._trainTrajectoryPath, 'rb') as f:
                data_dict = pickle.load(f)
        elif type=='val':
            with open(self._valTrajectoryPath, 'rb') as f:
                data_dict = pickle.load(f)
        elif type=='test':
            with open(self._testTrajectoryPath, 'rb') as f:
                data_dict = pickle.load(f)


        min_len = 1000000

        for key in data_dict.keys():
            data_dict[key] = np.array(data_dict[key])
            if data_dict[key].shape[0] < min_len:
                min_len = data_dict[key].shape[0]


        print('.........................min_len',min_len)



        for key in data_dict.keys():
            if n == 0:
                data['observations'] = np.expand_dims(np.array(data_dict[key])[:min_len-1, [1,2]], axis=0)
                data['actions'] = np.expand_dims(np.array(data_dict[key])[:min_len-1, [3,4]], axis=0)
                # data['tasks'] = np.array(data_np['tasks'][:,:-1,:])
                data['next_observations'] = np.expand_dims(np.array(data_dict[key])[1:min_len, [1,2]], axis=0)
                n = n + 1
                continue
            else:
                ## concatenate the data
                print(data['observations'].shape, data['observations'].shape)
                data['observations'] = np.concatenate(
                    (data['observations'], np.expand_dims(np.array(data_dict[key])[:min_len-1, [1,2]], axis=0)), axis=0)
                data['actions'] = np.concatenate((data['actions'], np.expand_dims(np.array(data_dict[key])[:min_len-1, [3,4]])), axis=0)
                # data['tasks'] = np.concatenate((data['tasks'], data_np['tasks'][:,:-1,:]), axis=0)
                data['next_observations'] = np.concatenate(
                    (data['next_observations'], np.expand_dims(np.array(data_dict[key])[1:min_len, [1,2]])), axis=0)

        obs = data['observations']
        act = data['actions']
        print('>>>>>>>>>>>>Processed Data Trajectories with shape<<<<<<<<<<<<<<<', obs.shape,
              act.shape)
        next_obs = data['next_observations']
        tasks = data['observations']
        return obs, act, next_obs, tasks


if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

