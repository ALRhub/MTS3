import torch
import numpy as np
from typing import Iterable, Tuple, List

nn = torch.nn


class Update(nn.Module):
    """
    The update of prior belief of MTS3 given an observation or a set of observations.
    Given a single observation we use the standard Kalman update equations in RKN paper.
    Given a set of observations we use the batched Kalman update equations derived in MTS3 paper.
    Observation tensor (single or multiple) should be of shape (batch_size, samples, lod).

    Note: We could as well use the batched Kalman update equations for a single observation. Mathematically they are equivalent.
    But computationally they are different. TODO: A detailed study on the computational complexity of the two approaches is needed.
    """

    def __init__(self, latent_obs_dim: int, memory: bool = True, config=None, dtype: torch.dtype = torch.float32):
        """
        :param latent_obs_dim: latent observation dimension
        :param memory: whether to use memory (H=[I,0] observation model) or not
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(Update, self).__init__()
        self._lod = latent_obs_dim
        self._mem = memory
        self._lsd = 2 * self._lod
        self.c = config
        self._dtype = dtype
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #   @torch.jit.script_method
    def forward(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor], obs: torch.Tensor,
                obs_var: torch.Tensor, obs_valid: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor, Iterable[torch.Tensor]]:
        """
        forward pass trough the cell. For proper recurrent model feed back outputs 3 and 4 (next prior belief at next
        time step
        :param prior_mean: prior mean at time t (batch_size, lsd)
        :param prior_cov: prior covariance at time t (batch_size, lsd) or a list of 3 tensors with (batch_size, lod)
        :param obs: observation at time t (batch_size, samples, lod)
        :param obs_var: observation variance at time t (batch_size, samples, lod)
        :param obs_valid: flag indicating whether observation at time t valid (batch_size, samples)
        :return: posterior mean at time t, posterior covariance at time t with same shape as prior_mean and prior_cov
        """

        post_mean, post_cov = self._masked_update(prior_mean, prior_cov, obs, obs_var, obs_valid)

        return post_mean, post_cov

    def _invert(self, S=Iterable[torch.Tensor]):
        """
        Invert a block matrix using eq in paper
        @param S: list with upper, lower and side part of either precision/covariance matrix
        @return: list with upper, lower and side part of inverted precision/covariance matrix
        """
        [s_u, s_l, s_s] = S
        d = (s_u * s_l - s_s * s_s)
        i_u = s_l / d
        i_s = -s_s / d
        i_l = s_u / d

        return [i_u, i_l, i_s]

    def _masked_update(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                        obs_mean: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor) -> Tuple[
        torch.Tensor, List[torch.Tensor]]:
        """Performs update step
        :param prior_mean: current prior state mean (batch_size, lsd)
        :param prior_cov: current prior state covariance (batch_size, lsd) or a list of 3 tensors with (batch_size, lod)
        :param obs_mean: current observation mean (batch_size, samples, lod)
        :param obs_var: current covariance mean (batch_size, samples, lod)
        :param obs_valid: flag indicating whether observation at time t valid (batch_size, samples)
        :return: current posterior state and covariance (batch_size, lsd) or a list of 3 tensors with (batch_size, lod)
        """
        ### assert that obs_mean and obs_var have 3 dimensions
        assert obs_mean.dim() == 3, "obs_mean should have 3 dimensions"
        assert obs_var.dim() == 3, "obs_var should have 3 dimensions"

        ####### Dealing with varying context using obs_valid flag. When observations are not valid we get a 0 mean and infinite variance embedding.
        if obs_valid is not None:
            obs_mean = obs_mean.where(obs_valid, torch.zeros(obs_mean.shape, device=self._device))
            obs_var = obs_var.where(obs_valid, np.inf * torch.ones(obs_mean.shape, device=self._device))

        ## use the masked mean and variance to compute the posterior
        if self._mem:
            if obs_mean.shape[1] == 1:
                # single observation
                # use the factorized kalman update equations in RKN paper

                # squeeze the second dimension
                obs_mean = obs_mean.squeeze(1)
                obs_var = obs_var.squeeze(1)

                ## Unpack prior covariance

                cov_u, cov_l, cov_s = prior_cov

                # compute kalman gain (eq 2 and 3 in paper)
                denominator = cov_u + obs_var
                q_upper = cov_u / denominator
                q_lower = cov_s / denominator

                # update mean (eq 4 in paper)
                residual = obs_mean - prior_mean[:, :self._lod]
                post_mean = prior_mean + torch.cat([q_upper * residual, q_lower * residual], -1)

                # update covariance (eq 5 -7 in paper)
                covar_factor = 1 - q_upper
                post_cov_u = covar_factor * cov_u
                post_cov_l = cov_l - q_lower * cov_s
                post_cov_s = covar_factor * cov_s

                ### pack cov into a list
                post_cov = [post_cov_u, post_cov_l, post_cov_s]  # posterior covariance matrix (batch_size, lsd)
            else:
                # multiple observations
                # use the factorized update equations in MTS3 paper

                #####Dividing mean into upper and lower parts
                prior_mean_u = prior_mean[:, :self._lod]  # upper prior mean (batch_size, lod)
                prior_mean_l = prior_mean[:, self._lod:]  # lower prior mean (batch_size, lod)

                prior_lam_u, prior_lam_l, prior_lam_s = self._invert(
                    prior_cov)  # upper, lower and side part of prior precision matrix each (batch_size, lod)

                ###### Update the posterior precision matrix
                cov_w_inv = 1 / obs_var  # inverse of observation covariances (batch_size, num_points, lod)
                post_lam = [prior_lam_u + torch.sum(cov_w_inv, dim=1), prior_lam_l,
                            prior_lam_s]  # posterior precision matrix (batch_size, lod)

                ####### Convert back to covariance matrix
                [post_cov_u, post_cov_l, post_cov_s] = self._invert(
                    post_lam)  # upper, lower and side part of posterior covariance matrix (batch_size, lod)

                #####Updating the lower and upper parts of mean using eq(1) to eq() in paper
                v = obs_mean - prior_mean_u[:, None, :]
                post_mu_u = prior_mean_u + post_cov_u * torch.sum(v * cov_w_inv,
                                                                    dim=1)  # upper part of posterior mean (batch_size, lod)
                post_mu_l = prior_mean_l + post_cov_s * torch.sum(v * cov_w_inv,
                                                                    dim=1)  # lower part of posterior mean (batch_size, lod)

                #### Pack and sent
                post_mean = torch.cat((post_mu_u, post_mu_l), dim=-1)  # posterior mean (batch_size, lsd)
                post_cov = [post_cov_u, post_cov_l, post_cov_s]  # posterior covariance matrix (batch_size, lsd)
        else:
            ### Simple Bayesian Aggregation Update from Volpp et al. 2020
            # create intial state
            initial_mean = prior_mean
            prior_cov_u, prior_cov_l, prior_cov_s = prior_cov
            ##concatenate the upper and lower
            initial_cov = torch.cat([prior_cov_u, prior_cov_l], dim=-1)

            v = obs_mean - initial_mean[:, None, :]
            cov_w_inv = 1 / obs_var
            # print('cov_w_inv', cov_w_inv.shape)
            cov_z_new = 1 / (1 / initial_cov + torch.sum(cov_w_inv, dim=1))
            # print('cov_z_new', cov_z_new.shape)
            mu_z_new = initial_mean + cov_z_new * torch.sum(cov_w_inv * v, dim=1)
            # print('mu_z_new', mu_z_new.shape)

            post_mean = torch.squeeze(mu_z_new)
            post_cov = torch.squeeze(cov_z_new)

            ### convert to upper and lower parts
            upper_length = int(self._lod / 2)
            post_cov_u = post_cov[:, :upper_length]
            post_cov_l = post_cov[:, upper_length:]
            post_cov_s = torch.zeros_like(post_cov_u)

            post_cov = [post_cov_u, post_cov_l, post_cov_s]

        if obs_valid is not None:
            ##Set post mean as prior mean if all obs_valid are false
            post_mean = post_mean.where(obs_valid.any(dim=1), prior_mean)
            ## Set post cov as prior cov if all observations are invalid

            post_cov = [post_cov_u.where(obs_valid.any(dim=1), prior_cov[0]),
                        post_cov_l.where(obs_valid.any(dim=1), prior_cov[1]),
                        post_cov_s.where(obs_valid.any(dim=1), prior_cov[2])]

        return post_mean, post_cov

