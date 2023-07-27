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

        self.trajPerTask = self.c.trajPerTask
        self._down_sample = self.c.downsample

        self.tar_type = self.c.tar_type
        self._obs_impu = self.c.obs_imp
        self._task_impu = self.c.task_imp

        self._split = OmegaConf.to_container(self.c.split)
        self._shuffle_split = self.c.shuffle_split

        self.num_training_sequences = self.c.num_training_sequences
        self.num_testing_sequences = self.c.num_testing_sequences
        self.episode_length = self.c.episode_length  # window length for a single hiprssm instance
        self.num_episodes = self.c.num_episodes # number of hiprssm instances
        self.normalizer = None
        self.standardize = self.c.standardize
        # data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid':obs_valid_batch}



    def get_statistics(self, data, dim):
        if len(data.shape) > 2:
            reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
            data = reshape(data);
        ## convert to float
        data = data.astype(np.float32)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def _load_trajectories(self):
        """
                Load from disk the obs, act, next_obs, task_labels
                """
        return NotImplementedError

    def _pre_process(self, obs, act, next_obs, tasks, rewards=None, test_obs=None, test_act=None, test_next_obs=None, test_tasks=None, test_rewards=None):
        ### Downsample
        #obs = obs[:,::self._down_sample,:]
        #act = act[:,::self._down_sample,:]
        #next_obs = next_obs[:,::self._down_sample,:]
        #tasks = tasks[:,::self._down_sample]
        #if rewards is not None:
        #    rewards = rewards[:,::self._down_sample,:]
        train_obs = obs
        train_act = act
        train_next_obs = next_obs
        train_tasks = tasks
        train_rewards = rewards
        if test_obs is None:
            # train test split
            if self._shuffle_split is not None:
                if rewards is not None:
                    train_obs, train_act, train_next_obs, train_tasks, train_rewards, \
                    test_obs, test_act, test_next_obs, test_tasks, test_rewards = self.shuffle_split(obs, act, next_obs, tasks, rewards)
                else:
                    train_obs, train_act, train_next_obs, train_tasks, \
                    test_obs, test_act, test_next_obs, test_tasks = self.shuffle_split(obs, act, next_obs, tasks)

            else:
                if rewards is not None:
                    train_obs, train_act, train_next_obs, train_tasks, train_rewards, test_obs, test_act, test_next_obs, test_tasks, test_rewards = self.train_test_split(obs, act, next_obs, tasks, rewards)
                else:
                    train_obs, train_act, train_next_obs, train_tasks, test_obs, test_act, test_next_obs, test_tasks = self.train_test_split(obs, act, next_obs, tasks)

        train_delta = train_next_obs - train_obs
        test_delta = test_next_obs - test_obs

        # get different statistics for state, actions, delta_state, delta_action and residuals which will be used for standardization
        mean_state_diff, std_state_diff = self.get_statistics(train_delta, dim=train_delta.shape[-1])
        mean_obs, std_obs = self.get_statistics(train_obs, dim=train_obs.shape[-1])
        mean_act, std_act = self.get_statistics(train_act, dim=train_act.shape[-1])
        if rewards is not None:
            mean_reward, std_reward = self.get_statistics(train_rewards, dim=train_rewards.shape[-1])
        self.normalizer = dict()
        self.normalizer['observations'] = [mean_obs, std_obs]
        self.normalizer['actions'] = [mean_act, std_act]
        self.normalizer['diff'] = [mean_state_diff, std_state_diff]
        if rewards is not None:
            self.normalizer['rewards'] = [mean_reward, std_reward]

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

            self.train_obs = normalize(train_obs, self.normalizer["observations"][0],
                                       self.normalizer["observations"][1])
            self.train_acts = normalize(train_act, self.normalizer["actions"][0],
                                        self.normalizer["actions"][1])

            self.test_obs = normalize(test_obs, self.normalizer["observations"][0],
                                      self.normalizer["observations"][1])
            self.test_acts = normalize(test_act, self.normalizer["actions"][0],
                                       self.normalizer["actions"][1])

            if self.tar_type == 'delta':
                self.train_targets = normalize(train_delta, self.normalizer["diff"][0],
                                               self.normalizer["diff"][1])
                self.test_targets = normalize(test_delta, self.normalizer["diff"][0],
                                              self.normalizer["diff"][1])
            else:
                self.train_targets = normalize(train_next_obs, self.normalizer["observations"][0],
                                               self.normalizer["observations"][1])
                self.test_targets = normalize(test_next_obs, self.normalizer["observations"][0],
                                              self.normalizer["observations"][1])

            if rewards is not None:
                self.train_rewards = normalize(train_rewards, self.normalizer["rewards"][0],
                                               self.normalizer["rewards"][1])
                self.test_rewards = normalize(test_rewards, self.normalizer["rewards"][0],
                                              self.normalizer["rewards"][1])


        else:
            self.train_obs = train_obs
            self.train_acts = train_act
            self.test_obs = test_obs
            self.test_acts = test_act
            if self.tar_type == 'delta':
                self.train_targets = train_delta
                self.test_targets = test_delta
            else:
                self.train_targets = train_next_obs
                self.test_targets = test_next_obs
            if rewards is not None:
                self.train_rewards = train_rewards
                self.test_rewards = test_rewards

        self.train_task_idx = train_tasks
        self.test_task_idx = test_tasks

        plot=False
        if plot:
            from matplotlib import pyplot as plt
            i=0
            for x,y in zip(self.train_task_idx,self.test_task_idx):
                plt.plot(x)
                #plt.plot(y)
                i=i+1
                plt.show()


        # Get random windows
        train_data_window = self._get_batch(train=True, use_reward=rewards)
        test_data_window = self._get_batch(train=False, use_reward=rewards)

        return train_data_window, test_data_window

    def _get_batch(self, train, use_reward=False):
        # Takes multiple paths and splits them into windows based on random locations within a trajectory
        if train:
            num_paths, len_path = self.train_obs.shape[:2]
            print("..............................", self.train_obs.shape)
            idx_path = np.random.randint(0, num_paths,
                                            size=self.num_training_sequences)  # task index, which gets mixed along the
            # process
            print(num_paths, len_path)
            idx_batch = np.random.randint(0, len_path - self.num_episodes*self.episode_length, size=self.num_training_sequences)
            obs_batch = np.array([self.train_obs[ip,
                                    ib:ib + self.num_episodes*self.episode_length, :]
                                    for ip, ib in zip(idx_path, idx_batch)])
            obs_batch = np.transpose(np.array(np.split(obs_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            act_batch = np.array([self.train_acts[ip,
                                    ib:ib + self.num_episodes*self.episode_length, :]
                                    for ip, ib in zip(idx_path, idx_batch)])
            act_batch = np.transpose(np.array(np.split(act_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            target_batch = np.array([self.train_targets[ip,
                                        ib:ib + self.num_episodes*self.episode_length, :]
                                        for ip, ib in zip(idx_path, idx_batch)])
            target_batch = np.transpose(np.array(np.split(target_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            t_idx_batch = np.array([self.train_task_idx[ip,
                                    ib:ib + self.num_episodes*self.episode_length]
                                    for ip, ib in zip(idx_path, idx_batch)])
            ## if self.train_rewards exists, then add it to the batch
            if use_reward:
                reward_batch = np.array([self.train_rewards[ip,
                                        ib:ib + self.num_episodes*self.episode_length]
                                        for ip, ib in zip(idx_path, idx_batch)])
                reward_batch = np.transpose(np.array(np.split(reward_batch,self.num_episodes,axis=1)),axes=[1,0,2])
            #print(t_idx_batch.shape,len(t_idx_batch.shape))
            if len(t_idx_batch.shape) == 2:
                t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes,axis=1)),axes=[1,0,2])
            else:
                t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes, axis=1)), axes=[1, 0, 2,3])

        else:
            np.random.seed(0)
            num_paths, len_path = self.test_obs.shape[:2]
            print("..............................", self.test_obs.shape)
            idx_path = np.random.randint(0, num_paths, size=self.num_testing_sequences)
            idx_batch = np.random.randint(0, len_path - self.num_episodes*self.episode_length, size=self.num_testing_sequences)
            #print(idx_path,"......", idx_batch)

            np.random.seed()
            obs_batch = np.array([self.test_obs[ip,
                                    ib:ib + self.num_episodes*self.episode_length, :]
                                    for ip, ib in zip(idx_path, idx_batch)])
            obs_batch = np.transpose(np.array(np.split(obs_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            act_batch = np.array([self.test_acts[ip,
                                    ib:ib + self.num_episodes*self.episode_length, :]
                                    for ip, ib in zip(idx_path, idx_batch)])
            act_batch = np.transpose(np.array(np.split(act_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            target_batch = np.array([self.test_targets[ip,
                                        ib:ib + self.num_episodes*self.episode_length, :]
                                        for ip, ib in zip(idx_path, idx_batch)])
            target_batch = np.transpose(np.array(np.split(target_batch,self.num_episodes,axis=1)),axes=[1,0,2,3])
            t_idx_batch = np.array([self.test_task_idx[ip,
                                    ib:ib + self.num_episodes*self.episode_length]
                                    for ip, ib in zip(idx_path, idx_batch)])
            if use_reward:
                reward_batch = np.array([self.test_rewards[ip,
                                        ib:ib + self.num_episodes*self.episode_length]
                                        for ip, ib in zip(idx_path, idx_batch)])
                reward_batch = np.transpose(np.array(np.split(reward_batch,self.num_episodes,axis=1)),axes=[1,0,2])

            if len(t_idx_batch.shape) == 2:
                t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes, axis=1)), axes=[1, 0, 2])
            else:
                t_idx_batch = np.transpose(np.array(np.split(t_idx_batch, self.num_episodes, axis=1)),
                                            axes=[1, 0, 2, 3])

        rs = np.random.RandomState(seed=42)
        obs_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], obs_batch.shape[2], 1) < 1 - self._obs_impu
        #TODO: Episodic Case
        #if self.c.episodic:
        #    obs_valid_batch[:, :, :5] = True
        #else:
        obs_valid_batch[:, 0, :5] = True


        if self._long_term_pred > 1:
            #set obs_valid for meta-step to False when task_valid is false (to verify)
            #TODO Verify
            task_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], 1) < 1 - self._task_impu
            task_valid_batch[:, -self._long_term_pred:] = False
            task_valid_batch[:, :1] = True
            #obs_valid_batch[:, -self._long_term_pred:, :] = False
        else:
            task_valid_batch = rs.rand(obs_batch.shape[0], obs_batch.shape[1], 1) < 1 - self._task_impu
            task_valid_batch[:,:1] = True


        if use_reward:
            data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid': obs_valid_batch, "task_valid": task_valid_batch, "reward": reward_batch,
                        'normalization': self.normalizer,
                        'task_index': np.mean(t_idx_batch,-1)}
        else:
            data_windows = {'obs': obs_batch, 'act': act_batch, 'target': target_batch, 'obs_valid': obs_valid_batch, "task_valid": task_valid_batch,
                        'normalization': self.normalizer,
                        'task_index': np.mean(t_idx_batch,-1)}  ###CLEVER TRICK %trajPerTask
        # TODO for the target(second half) initialize few things to True
        return data_windows

    def shuffle_split(self, obs, act, target, task_idx, reward=None):
        """
        Shuffle and split the data into training and testing sets.
        """
        print(obs.shape[0], act.shape[0], target.shape[0], task_idx.shape[0])
        assert obs.shape[0] == act.shape[0] == target.shape[0]

        ## Get train and test indices after shuffling and splitting
        dataset_size = obs.shape[0]
        indices = np.arange(dataset_size)
        print(indices, dataset_size)
        np.random.shuffle(indices)
        print(indices)
        split_idx = int(dataset_size * self._shuffle_split)
        idx_train = indices[:split_idx]
        idx_test = indices[split_idx:]

        print('Train Test Split Ratio', self._shuffle_split)
        assert len(idx_train) + len(idx_test) <= dataset_size

        if reward is not None:
            assert reward.shape[0] == obs.shape[0]
            return obs[idx_train], act[idx_train], target[idx_train], task_idx[idx_train], reward[idx_train], \
                obs[idx_test], act[idx_test], target[idx_test], task_idx[idx_test], reward[idx_test]
        else:
            return obs[idx_train], act[idx_train], target[idx_train], task_idx[idx_train], \
                obs[idx_test], act[idx_test], target[idx_test], task_idx[idx_test]

    def train_test_split(self, obs, act, delta, grad, reward=None):
        print(obs.shape[0],act.shape[0],delta.shape[0])
        assert obs.shape[0] == act.shape[0] == delta.shape[0]
        assert isinstance(self._split, list) or isinstance(self._split, float)
        episodes = obs.shape[0]

        assert len(self._split) == 2
        idx_train = self._split[0]
        idx_test = self._split[1]
        print('Training Indices:', idx_train, 'Testing Indices:', idx_test)

        # idx_test = [8,9]

        assert len(idx_train) + len(idx_test) <= episodes

        if reward is not None:
            assert reward.shape[0] == obs.shape[0]
            return obs[idx_train], act[idx_train], delta[idx_train], grad[idx_train], reward[idx_train], \
                        obs[idx_test], act[idx_test], delta[idx_test], grad[idx_test], reward[idx_test]
        else:
            return obs[idx_train], act[idx_train], delta[idx_train], grad[idx_train], \
                        obs[idx_test], act[idx_test], delta[idx_test], grad[idx_test]





if __name__ == '__main__':
    dataFolder = os.getcwd() + '/dataFolder/MobileRobot/sin2/'
    # self._trajectoryPath = self._dataFolder + 'HalfCheetahEnv_6c2_cripple.pickle'
    trajectoryPath = dataFolder + 'ts_002_50x2000.npz'
    data = np.load(trajectoryPath)
    print(data.keys())
    print(np.sin(data['orn_euler']))
    print(np.cos(data['orn_euler']))
    print(data['orn_euler'])

