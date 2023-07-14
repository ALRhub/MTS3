# TODO: collect valid flags too ??
# TODO: go through the code once again
# TODO: check if update and marginalization is correct

import torch
from omegaconf import DictConfig, OmegaConf
from utils.TimeDistributed import TimeDistributed
from utils.vision.torchAPI import Reshape
from agent.worldModels.SensorEncoders.propEncoder import Encoder
from gaussianTransformations.gaussian_marginalization import Predict
from gaussianTransformations.gaussian_conditioning import Update
from agent.worldModels.Decoders.propDecoder import SplitDiagGaussianDecoder 
from utils.dataProcess import norm, denorm
import numpy.random as rd

nn = torch.nn

'''
Tip: in config self.ltd = lod 
in context_predict ltd is doubled
in task_predict ltd is considered lod and lsd is doubled inside
'''
class MTS3(nn.Module):
    """
    MTS3 model
    Inference happen in such a way that first episode is used for getting an intial task posterioer and then the rest of the episodes are used for prediction by the worker
    Maybe redo this logic based on original implementation or use a different method that helps control too ??
    """

    def __init__(self, time_scale_multiplier, obs_dim=None, action_dim=None, inp_shape=None, config=None, use_cuda_if_available: bool = True):
        """
        @param obs_dim: dimension of observations to train on
        @param action_dim: dimension of control signals
        @param inp_shape: shape of the input observations
        @param config: config dict
        @param dtype:
        @param use_cuda_if_available:
        """
        super(MTS3, self).__init__()
        if config == None:
            raise ValueError("config cannot be None, pass an omegaConf File")
        else:
            self.c = config
        self.H = time_scale_multiplier
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._lod = self.c.latent_obs_dim
        self._pixel_obs = self.c.mts3.pixel_obs ##TODO: config
        self._decode_reward = self.c.mts3.decode.reward ##TODO: config and it basically initializes the reward decoder 
        self._decode_obs = self.c.mts3.decode.obs ##TODO: config and it basically initializes the obs decoder



        ### Define the encoder and decoder
        obsEnc = Encoder(self._obs_dim, self.c.obsEnc_net_hidden_units, output_normalization=self._enc_out_normalization, activation=self.c.variance_act) ## TODO: config
        self._obsEnc = TimeDistributed(obsEnc, num_outputs=2).to(self._device)

        absObsEnc = Encoder(self._obs_dim, self.c.absObs_enc_net_hidden_units, output_normalization=self._enc_out_normalization, activation=self.c.variance_act) ## TODO: config
        self._absObsEnc = TimeDistributed(absObsEnc, num_outputs=2).to(self._device)

        absActEnc = Encoder(self._action_dim, self.c.absAct_enc_net_hidden_units, output_normalization=self._enc_out_normalization, activation=self.c.variance_act) ## TODO: config
        self._absActEnc = TimeDistributed(absActEnc, num_outputs=2).to(self._device)

        obsDec = SplitDiagGaussianDecoder(out_dim=self._lod, activation=self.c.variance_act) ## TODO: config
        self._obsDec = TimeDistributed(obsDec, num_outputs=2).to(self._device)

        if decode_reward:
            rewardDec = SplitDiagGaussianDecoder(out_dim=1, activation=self.c.variance_act) ## TODO: config
            self._rewardDec = TimeDistributed(rewardDec, num_outputs=2).to(self._device)



        ### Define the gaussian layers for both levels
        self._state_predict = Predict(latent_obs_dim=self._lod, act_dim=self._lad, msw_flag = 2, config=self.c) ## initiate worker marginalization layer for state prediction
        self._task_predict = Predict(latent_obs_dim=self._lod, act_dim=self._lad, msw_flag = 0, config=self.c) ## initiate manager marginalization layer for task prediction

        self._obsUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c) ## memory is true
        self._taskUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c) ## memory is true

        self._action_Infer = Update(latent_obs_dim=self._lod, memory = False, config = self.c) ## memory is false

    def _intialize_mean_covar(self, batch_size, learn=False):
        if learn:
            pass

        else:
            init_state_covar_ul = self.c.task_dynamics_mem.initial_state_covar * torch.ones(batch_size, self._lsd)

            initial_mean = torch.zeros(batch_size, self._lsd).to(self._device)
            icu = init_state_covar_ul[:, :self._ltd].to(self._device)
            icl = init_state_covar_ul[:, self._ltd:].to(self._device)
            ics = torch.ones(1, self._ltd).to(self._device)

            initial_cov = [icu, icl, ics]

        return initial_mean, initial_cov


    def _create_time_embedding(self, batch_size, time_steps):
        """
        Creates a time embedding for the given batch size and time steps"""
        time_embedding = torch.zeros(batch_size, time_steps, 1).to(self._device)
        for i in range(time_steps):
            time_embedding[:, i, :] = i / time_steps
        return time_embedding
    
    def forward(self, obs_seqs, action_seqs, obs_valid_seqs, task_valid_seqs, decode_obs=True, decode_reward=False, train=False):
        '''
        obs_seqs: sequences of timeseries of observations (batch x time x obs_dim)
        action_seqs: sequences of timeseries of actions (batch x time x obs_dim)
        obs_valid_seqs: sequences of timeseries of actions (batch x time)
        task_valid_seqs: sequences of timeseries of actions (batch x task)
        '''
        ##################################### Manager ############################################
        # prepare list for return
        prior_task_mean_list = []
        prior_task_cov_list = []

        post_task_mean_list = []
        post_task_cov_list = []

        abs_act_list = []

        ### initialize mean and covariance for the first time step
        task_prior_mean, task_prior_cov = self._intialize_mean_covar(obs_seqs.shape[0], learn=False)


        ### loop over individual episodes in steps of H (Coarse time scale / manager)
        for k in range(0, obs_seqs.shape[1], self.H):
            ### encode the observation set with time embedding to get abstract observation
            time_embedding = self._create_time_embedding(obs_seqs.shape[0], self.H)
            current_obs_seqs = obs_seqs[:, k:k+self.H, :]
            beta_k_mean, beta_k_var = self._obsEnc(torch.cat([current_obs_seqs, time_embedding], dim=-1))

            ### get task valid for the current episode
            task_valid = task_valid_seqs[:, k, :] ##TODO: how to get this in the function that calls this function?
            ### update the task posterior with beta_current
            task_post_mean, task_post_cov = self._taskUpdate(task_prior_mean, task_prior_cov, beta_k_mean, beta_k_var, task_valid)
            
            ### encode the action set to get abstract action with time embedding
            current_act_seqs = action_seqs[:, k:k+self.H, :]
            abs_act_mean, abs_act_var = self._absActEnc(torch.cat([current_act_seqs, time_embedding], dim=-1))

            ### predict the next task mean and covariance using manager marginalization layer 
            ### with the current task posterior and abstract action as causal factors
            mean_list_causal_factors = [task_post_mean, abs_act_mean]
            cov_list_causal_factors = [task_post_cov, abs_act_var]
            task_next_mean, task_next_cov = self._task_predict(mean_list_causal_factors, cov_list_causal_factors)

            ### update the task prior
            task_prior_mean, task_prior_cov = task_next_mean, task_next_cov

            ### append the task mean and covariance to the list
            prior_task_mean_list.append(task_prior_mean)
            prior_task_cov_list.append(task_prior_cov)
            post_task_mean_list.append(task_post_mean)
            post_task_cov_list.append(task_post_cov)
            abs_act_list.append(abs_act_mean)

        ### stack the list to get the final tensors
        prior_task_means = torch.stack(prior_task_mean_list, dim=1)
        prior_task_covs = torch.stack(prior_task_cov_list, dim=1)
        post_task_means = torch.stack(post_task_mean_list, dim=1)
        post_task_covs = torch.stack(post_task_cov_list, dim=1)
        abs_acts = torch.stack(abs_act_list, dim=1)

        ### get the number of episodes from the length of prior_task_mean
        num_episodes = prior_task_mean.shape[1]

        
        ##################################### Worker ############################################
        ### using the task prior, predict the observation mean and covariance for fine time scale / worker
        ### create a meta_list of prior and posterior states

        global_state_prior_mean_list = []
        global_state_prior_cov_list = []
        global_state_post_mean_list = []
        global_state_post_cov_list = []

        for k in range(0,num_episodes): ## first episode is considered too to predict (but usually ignored in evaluation)
            ### create list of state mean and covariance 
            prior_state_mean_list = []
            prior_state_cov_list = []
            post_state_mean_list = []
            post_state_cov_list = []

            if k==0:
                state_prior_mean, state_prior_cov = self._intialize_mean_covar(obs_seqs.shape[0], learn=False)
            else:
                ### detach the last episode's prior state mean and covariance from the graph 
                ### so that in the worker gradients are not backpropagated intra-episode
                ### TODO: recheck if this is correct
                state_prior_mean = state_prior_mean.detach()
                state_prior_cov = [cov.detach() for cov in state_prior_cov]


            ### get the task post for the current episode (here the assumption is that when observations are missing the task valid flag keeps posterior = prior)
            task_mean = post_task_means[:, k, :]
            task_cov = post_task_covs[:, k, :]

            ### get the obs, action for the current episode
            current_obs_seqs = obs_seqs[:, k*self.H:(k+1)*self.H, :]
            current_act_seqs = action_seqs[:, k*self.H:(k+1)*self.H, :]
            current_obs_valid_seqs = obs_valid_seqs[:, k*self.H:(k+1)*self.H, :]
            current_task_valid = task_valid_seqs[:, k, :]

            ### if task valid is 0, then make all current_obs_valid 0 (making sure the entire episode is masked out)
            current_obs_valid_seqs = current_obs_valid_seqs * current_task_valid.unsqueeze(1).repeat(1, self.H)

            for t in range(H):
                ### encode the observation (no time embedding)
                current_obs = current_obs_seqs[:, t, :]
                ## expand dims to make it compatible with the encoder
                current_obs = torch.unsqueeze(current_obs, dim=1)
                obs_mean, obs_var = self._obsEnc(current_obs)

                ### update the state posterior
                current_obs_valid = current_obs_valid_seqs[:, t, :]
                ## expand dims to make it compatible with the encoder
                current_obs_valid = torch.unsqueeze(current_obs_valid, dim=1)
                state_post_mean, state_post_cov = self._stateUpdate(state_prior_mean, state_prior_cov, obs_mean, obs_var, current_obs_valid)

                ### predict the next state mean and covariance using the marginalization layer for worker
                current_act = current_act_seqs[:, t, :]
                mean_list_causal_factors = [state_post_mean, current_act, task_mean]
                cov_list_causal_factors = [state_post_cov, task_cov]
                state_next_mean, state_next_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors)

                ### update the state prior
                state_prior_mean, state_prior_cov = state_next_mean, state_next_cov  ### this step also makes sure every episode 
                                                                                        ### starts with the prior of the previous episode

                ### append the state mean and covariance to the list
                prior_state_mean_list.append(state_prior_mean)
                prior_state_cov_list.append(state_prior_cov)
                post_state_mean_list.append(state_post_mean)
                post_state_cov_list.append(state_post_cov)

            ### stack the list to get the final tensors
            prior_state_means = torch.stack(prior_state_mean_list, dim=1)
            prior_state_covs = torch.stack(prior_state_cov_list, dim=1)
            post_state_means = torch.stack(post_state_mean_list, dim=1)
            post_state_covs = torch.stack(post_state_cov_list, dim=1)

            ### append the state mean and covariance to the list
            global_state_prior_mean_list.append(prior_state_means)
            global_state_prior_cov_list.append(prior_state_covs)
            global_state_post_mean_list.append(post_state_means)
            global_state_post_cov_list.append(post_state_covs)

        ### stack the list to get the final tensors
        global_state_prior_means = torch.stack(global_state_prior_mean_list, dim=1)
        global_state_prior_covs = torch.stack(global_state_prior_cov_list, dim=1)
        global_state_post_means = torch.stack(global_state_post_mean_list, dim=1)
        global_state_post_covs = torch.stack(global_state_post_cov_list, dim=1)

        ### decode the state to get the observation mean and covariance ##TODO: do it here ?? or outside ???
        if decode_obs:
            pred_obs_means, pred_obs_covs = self._obsDec(global_state_prior_means, global_state_prior_covs)
        if decode_reward:
            pred_reward_means, pred_reward_covs = self._rewardDec(global_state_prior_means, global_state_prior_covs)
        
        ## TODO: decode value and policy (for what abstractions??)
            
        return pred_obs_means, pred_obs_covs, prior_task_means, prior_task_covs, post_task_means, post_task_covs, abs_acts
