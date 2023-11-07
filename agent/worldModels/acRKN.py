# TODO: collect valid flags too ??
# TODO: go through the code once again
# TODO: check if update and marginalization is correct

import torch
from agent.worldModels.SensorEncoders.propEncoder import Encoder
from agent.worldModels.gaussianTransformations.gaussian_marginalization import Predict
from agent.worldModels.gaussianTransformations.gaussian_conditioning import Update
from agent.worldModels.Decoders.propDecoder import SplitDiagGaussianDecoder

nn = torch.nn

'''
Tip: in config self.ltd = lod 
in context_predict ltd is doubled
in task_predict ltd is considered lod and lsd is doubled inside
'''
class acRKN(nn.Module):
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
        super(acRKN, self).__init__()
        if config == None:
            raise ValueError("config cannot be None, pass an omegaConf File")
        else:
            self.c = config
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._obs_shape = input_shape
        self._action_dim = action_dim
        self._lod = self.c.acrkn.latent_obs_dim
        self._lsd = 2*self._lod

        self._pixel_obs = self.c.acrkn.pixel_obs ##TODO: config
        self._decode_reward = self.c.acrkn.decode.reward ##TODO: config and it basically initializes the reward decoder 
        self._decode_obs = self.c.acrkn.decode.obs ##TODO: config and it basically initializes the obs decoder



        ### Define the encoder and decoder
        obsEnc = Encoder(self._obs_shape[-1], self._lod, self.c.acrkn.worker.obs_encoder) ## TODO: config
        self._obsEnc = obsEnc.to(self._device)

        obsDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=self._obs_shape[-1], config=self.c.acrkn.worker.obs_decoder) ## TODO: config
        self._obsDec = obsDec.to(self._device)

        if self._decode_reward:
            rewardDec = SplitDiagGaussianDecoder(latent_obs_dim=self._lod, out_dim=1, config=self.c.acrkn.worker.reward_decoder) ## TODO: config
            self._rewardDec = rewardDec.to(self._device)


        ### Define the gaussian layers for both levels
        self._state_predict = Predict(latent_obs_dim=self._lod, act_dim=self._action_dim, hierarchy_type = "ACRKN", config=self.c.acrkn.worker) ## initiate worker marginalization layer for state prediction
        self._obsUpdate = Update(latent_obs_dim=self._lod, memory = True, config = self.c) ## memory is true
        
    def _intialize_mean_covar(self, batch_size, learn=False):
        if learn:
            pass

        else:
            init_state_covar_ul = self.c.acrkn.initial_state_covar * torch.ones(batch_size, self._lsd)

            initial_mean = torch.zeros(batch_size, self._lsd).to(self._device)
            icu = init_state_covar_ul[:, :self._lod].to(self._device)
            icl = init_state_covar_ul[:, self._lod:].to(self._device)
            ics = torch.ones(1, self._lod).to(self._device)

            initial_cov = [icu, icl, ics]

        return initial_mean, initial_cov
    
    def forward(self, obs_seqs, action_seqs, obs_valid_seqs, decode_obs=True, decode_reward=False, train=False):
        '''
        obs_seqs: sequences of timeseries of observations (batch x time x obs_dim)
        action_seqs: sequences of timeseries of actions (batch x time x obs_dim)
        obs_valid_seqs: sequences of timeseries of actions (batch x time)
        task_valid_seqs: sequences of timeseries of actions (batch x task)
        '''
        ##################################### Only Worker (with no task conditioning) ############################################
        ### using the task prior, predict the observation mean and covariance for fine time scale / worker
        ### create a meta_list of prior and posterior states
        num_episodes = 1

        state_prior_mean_init, state_prior_cov_init = self._intialize_mean_covar(obs_seqs.shape[0], learn=False)

        for k in range(0,num_episodes):
            if k==0:
                state_prior_mean = state_prior_mean_init
                state_prior_cov = state_prior_cov_init
            ### create list of state mean and covariance 
            prior_state_mean_list = []
            prior_state_cov_list = []
            post_state_mean_list = []
            post_state_cov_list = []

            for t in range(obs_seqs.shape[1]):
                ### encode the observation (no time embedding)
                current_obs = obs_seqs[:, t, :]

                ## expand dims to make it compatible with the encoder
                current_obs = torch.unsqueeze(current_obs, dim=1)
                obs_mean, obs_var = self._obsEnc(current_obs)

                ### update the state posterior
                current_obs_valid = obs_valid_seqs[:, t, :]

                ## expand dims to make it compatible with the encoder
                current_obs_valid = torch.unsqueeze(current_obs_valid, dim=1)
                state_post_mean, state_post_cov = self._obsUpdate(state_prior_mean, state_prior_cov, obs_mean, obs_var, current_obs_valid)

                ### predict the next state mean and covariance using the marginalization layer for ACRKN
                current_act = action_seqs[:, t, :]
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

        ### decode the state to get the observation mean and covariance ##TODO: do it here ?? or outside ???
        if self._decode_obs:
            pred_obs_means, pred_obs_covs = self._obsDec(prior_state_means, prior_state_covs)
        if self._decode_reward:
            pred_reward_means, pred_reward_covs = self._rewardDec(prior_state_means, prior_state_covs)

        return pred_obs_means, pred_obs_covs
