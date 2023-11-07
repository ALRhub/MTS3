# TODO: collect valid flags too ??
# TODO: go through the code once again
# TODO: check if update and marginalization is correct

import torch
from omegaconf import DictConfig, OmegaConf
from utils.vision.torchAPI import Reshape
from agent.worldModels.SensorEncoders.propEncoder import Encoder
from agent.worldModels.gaussianTransformations.gaussian_marginalization import Predict
from agent.worldModels.gaussianTransformations.gaussian_conditioning import Update
from agent.worldModels.Decoders.propDecoder import SplitDiagGaussianDecoder 
from utils.dataProcess import norm, denorm
import numpy.random as rd

nn = torch.nn

'''
Tip: in config self.ltd = lod 
in context_predict ltd is doubled
in task_predict ltd is considered lod and lsd is doubled inside
'''
class RKN(nn.Module):
    """
    MTS3 model
    Inference happen in such a way that first episode is used for getting an intial task posterioer and then the rest of the episodes are used for prediction by the worker
    Maybe redo this logic based on original implementation or use a different method that helps control too ??
    """

    def __init__(self, input_shape=None, action_dim=None, config=None, use_cuda_if_available: bool = True):
        """
        @param obs_dim: dimension of observations to train on
        @param action_dim: dimension of control signals
        @param inp_shape: shape of the input observations
        @param config: config dict
        @param dtype:
        @param use_cuda_if_available:
        """
        super(RKN, self).__init__()
        if config == None:
            raise ValueError("config cannot be None, pass an omegaConf File")
        else:
            self.c = config
        self.H = self.c.mts3.time_scale_multiplier
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._obs_shape = input_shape
        self._action_dim = action_dim
        self._lod = self.c.mts3.latent_obs_dim
        self._lsd = 2*self._lod
        self._time_embed_dim = self.c.mts3.manager.abstract_obs_encoder.time_embed.dim
        assert self._time_embed_dim == self.c.mts3.manager.abstract_act_encoder.time_embed.dim, \
                                            "Time Embedding Dimensions for obs and act encoder should be same"
        self._pixel_obs = self.c.mts3.pixel_obs ##TODO: config
        self._decode_reward = self.c.mts3.decode.reward ##TODO: config and it basically initializes the reward decoder 
        self._decode_obs = self.c.mts3.decode.obs ##TODO: config and it basically initializes the obs decoder



        ### Define the encoder and decoder
        obsEnc = Encoder(self._obs_shape[-1], self._lod, self.c.mts3.worker.obs_encoder) ## TODO: config
        self._obsEnc = obsEnc.to(self._device)

        obsDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=self._obs_shape[-1], config=self.c.mts3.worker.obs_decoder) ## TODO: config
        self._obsDec = obsDec.to(self._device)

        if self._decode_reward:
            rewardDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=1, config=self.c.mts3.worker.reward_decoder) ## TODO: config
            self._rewardDec = rewardDec.to(self._device)



        ### Define the gaussian layers for both levels
        self._state_predict = Predict(latent_obs_dim=self._lod, act_dim=self._action_dim, hierarchy_type = "ACRKN", config=self.c.mts3.worker) ## initiate worker marginalization layer for state prediction
        self._obsUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c) ## memory is true
        
    def _intialize_mean_covar(self, batch_size, learn=False):
        if learn:
            pass

        else:
            init_state_covar_ul = self.c.mts3.initial_state_covar * torch.ones(batch_size, self._lsd)

            initial_mean = torch.zeros(batch_size, self._lsd).to(self._device)
            icu = init_state_covar_ul[:, :self._lod].to(self._device)
            icl = init_state_covar_ul[:, self._lod:].to(self._device)
            ics = torch.ones(1, self._lod).to(self._device)

            initial_cov = [icu, icl, ics]

        return initial_mean, initial_cov


    def _create_time_embedding(self, batch_size, time_steps):
        """
        Creates a time embedding for the given batch size and time steps
        of the form (batch_size, time_steps, 1)"""
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
        ##################################### Manager Being Decoded ############################################
        ### using the task prior, predict the observation mean and covariance for fine time scale / worker
        ### create a meta_list of prior and posterior states

        global_state_prior_mean_list = []
        global_state_prior_cov_list = []
        global_state_post_mean_list = []
        global_state_post_cov_list = []

        state_prior_mean_init, state_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0], learn=False)

        for k in range(0,num_episodes): ## first episode is considered too to predict (but usually ignored in evaluation)
            if k==0:
                state_prior_mean = state_prior_mean_init
                state_prior_cov = state_prior_cov_init
            ### create list of state mean and covariance 
            prior_state_mean_list = []
            prior_state_cov_list = []
            post_state_mean_list = []
            post_state_cov_list = []

            
            ### if task valid is 0, then make all current_obs_valid 0 (making sure the entire episode is masked out)
            current_obs_valid_seqs = current_obs_valid_seqs * current_task_valid.unsqueeze(1).repeat(1, self.H, 1)

            for t in range(obs_seqs.shape[1]):
                ### encode the observation (no time embedding)
                current_obs = current_obs_seqs[:, t, :]
                ## expand dims to make it compatible with the encoder
                current_obs = torch.unsqueeze(current_obs, dim=1)
                obs_mean, obs_var = self._obsEnc(current_obs)

                ### update the state posterior
                current_obs_valid = current_obs_valid_seqs[:, t, :]
                ## expand dims to make it compatible with the encoder
                current_obs_valid = torch.unsqueeze(current_obs_valid, dim=1)
                state_post_mean, state_post_cov = self._obsUpdate(state_prior_mean, state_prior_cov, obs_mean, obs_var, current_obs_valid)

                ### predict the next state mean and covariance using the marginalization layer for worker
                current_act = current_act_seqs[:, t, :]
                mean_list_causal_factors = [state_post_mean, current_act]
                cov_list_causal_factors = [state_post_cov]
                state_next_mean, state_next_cov = self._state_predict(mean_list_causal_factors, cov_list_causal_factors)

                ### update the state prior
                state_prior_mean, state_prior_cov = state_next_mean, state_next_cov  ### this step also makes sure every episode 
                                                                                        ### starts with the prior of the previous episode

                ### concat 
                ### append the state mean and covariance to the list
                prior_state_mean_list.append(state_prior_mean)
                prior_state_cov_list.append(torch.cat(state_prior_cov, dim=-1))
                post_state_mean_list.append(state_post_mean)
                post_state_cov_list.append(torch.cat(state_post_cov, dim=-1))
            
            ## detach the state prior mean and covariance to make sure the next episode starts with the prior of the previous episode
            state_prior_mean = state_prior_mean.detach()
            state_prior_cov = [cov.detach() for cov in state_prior_cov]

            ### stack the list to get the final tensors
            prior_state_means = torch.stack(prior_state_mean_list, dim=1)
            prior_state_covs = torch.stack(prior_state_cov_list, dim=1)
            post_state_means = torch.stack(post_state_mean_list, dim=1)
            post_state_covs = torch.stack(post_state_cov_list, dim=1)


        ### decode the state to get the observation mean and covariance ##TODO: do it here ?? or outside ???
        if self._decode_obs:
            pred_obs_means, pred_obs_covs = self._obsDec(prior_state_means, prior_state_covs)
        if self._decode_reward:
            pred_reward_means, pred_reward_covs = self._rewardDec(  prior_state_means, prior_state_covs)
        
        ## TODO: decode value and policy (for what abstractions??)
            
        return pred_obs_means, pred_obs_covs
