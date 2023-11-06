import sys
import os

sys.path.append('.')
from dataFolder.dataDpssm import metaData
import numpy as np
import pickle
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


class metaCheetahData(metaData):
    def __init__(self, data_cfg=None):
        super(metaCheetahData, self).__init__(data_cfg)
        if data_cfg is None:
            raise Exception('Please specify a valid Confg for data')
        else:
            self.c = data_cfg

        if self.c.type == '0':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0/'
            self._trajectoryPath = None
        elif self.c.type == '1':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_001/'
            self._trajectoryPath = None
        elif self.c.type == '2':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_005/'
            self._trajectoryPath = None
        elif self.c.type == '3':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_01/'
            self._trajectoryPath = None
        elif self.c.type == '1_3':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_001/'
            self._trajectoryPath = self._dataFolder + 'experience_3.pickle'
        elif self.c.type == '2_2':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_005/'
            self._trajectoryPath = self._dataFolder + 'experience_2.pickle'
        elif self.c.type == '2_3':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_005/'
            self._trajectoryPath = self._dataFolder + 'experience_3.pickle'
        elif self.c.type == '3_1':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_01/'
            self._trajectoryPath = self._dataFolder + 'experience_1.pickle'
        elif self.c.type == '3_2':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_01/'
            self._trajectoryPath = self._dataFolder + 'experience_2.pickle'
        elif self.c.type == '3_3':
            self._dataFolder = get_original_cwd() + '/dataFolder/CheetahWind/new/0_01/'
            self._trajectoryPath = self._dataFolder + 'experience_3.pickle'
        elif self.c.type == 'd4rl':
            self._dataFolder = get_original_cwd() + '/dataFolder/d4rl/'
            self._trajectoryPath = self._dataFolder + 'halfcheetah-medium-v0.pkl'
        else:
            ### Raise python Error enter valid type
            raise Exception('Please specify a valid type for data')

            
        #self._trajectoryPath = '/home/vshaj/CLAS/meta-ssm/dataFolder/MujocoFolder/dyn_model.pickle'

        self.trajPerTask = self.c.trajPerTask

        self.tar_type = self.c.tar_type

        self._split = OmegaConf.to_container(self.c.split)
        self._shuffle_split = self.c.shuffle_split

        self.num_training_sequences = self.c.num_training_sequences
        self.num_testing_sequences = self.c.num_testing_sequences
        self.episode_length = self.c.episode_length  # window length for a single hiprssm instance
        self.num_episodes = self.c.num_episodes # number of hiprssm instances
        self.normalizer = None
        self.standardize = self.c.standardize
        obs, act, next_obs, tasks, rewards = self._load_trajectories()
        self.train_windows, self.test_windows = self._pre_process(obs, act, next_obs, tasks)
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}


    def _load_trajectories(self):
        # collect obs, act, next states
        data = {'observations': [], 'actions': [], 'next_observations': [], 'rewards': []}
        if self._trajectoryPath is None:
            n=0
            for filename in os.listdir(self._dataFolder):
                if filename.endswith(".pickle"):
                    self._trajectoryPath = self._dataFolder + filename
                    with open(self._trajectoryPath, 'rb') as f:
                        data_np = pickle.load(f)
                        if n == 0:
                            data['observations'] = np.array(data_np['observations'][:, :-1, :9])
                            data['actions'] = np.array(data_np['actions'][:, :-1, :6])
                            # data['tasks'] = np.array(data_np['tasks'][:,:-1,:])
                            data['next_observations'] = np.array(data_np['observations'][:, 1:, :9])
                            data['rewards'] = np.array(data_np['rewards'][:, :-1])
                            n=n+1
                            continue
                        else:
                            ## concatenate the data
                            print(data_np['observations'].shape, data['observations'].shape)
                            data['observations'] = np.concatenate((data['observations'], data_np['observations'][:, :-1, :9]), axis=0)
                            data['actions'] = np.concatenate((data['actions'], data_np['actions'][:, :-1, :6]), axis=0)
                            # data['tasks'] = np.concatenate((data['tasks'], data_np['tasks'][:,:-1,:]), axis=0)
                            data['next_observations'] = np.concatenate((data['next_observations'], data_np['observations'][:, 1:, :9]), axis=0)
                            data['rewards'] = np.concatenate((data['rewards'], data_np['rewards'][:, :-1]), axis=0)
        else:
            with open(self._trajectoryPath, 'rb') as f:
                data_np = pickle.load(f)
            print('>>>>>>>>>>>>Loaded Data Trajectories with shape<<<<<<<<<<<<<<<', data_np['observations'].shape)
            # dim = np.r_[0:15, 30:41]
            data['observations'] = data_np['observations'][:,:-1,:9]
            data['actions'] = data_np['actions'][:,:-1,:6]
            #data['tasks'] = data_np['tasks'][:,:-1,:]
            data['next_observations'] = data_np['observations'][:,1:,:9]
            data['rewards'] = data_np['rewards'][:,:-1]
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

