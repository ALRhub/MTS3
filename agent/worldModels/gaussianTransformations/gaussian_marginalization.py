import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import Iterable, Tuple, List

nn = torch.nn

def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]

def gaussian_linear_transform(tm, mean: torch.Tensor, covar:torch.Tensor, mem=True):
    ##Eliminated the eed for memory condition as structure of mean and cov unified
    """
    Performs marginalization of a gaussian distribution. This uses efficient sparse matrix multiplications,
    especially for the covariance matrix. This makes use of the block structure of the covariance matrix and the fact
    that we are only interested in the diagonal elements of the blocks of the resulting covariance matrix.
    :param tm: list of transition matrices
    :param mean: prior mean
    :param covar: prior covariance
    :return: next prior mean and covariance
    """

    # predict next prior mean
    # TODO: recheck separate cases for with and without memory
    if mem:
        obs_dim = int(mean.shape[-1]/2)
        mu = mean[:, :obs_dim]
        ml = mean[:, obs_dim:]
        #[tm11, tm12, tm21, tm22] = [t.repeat((mu.shape[0], 1, 1)) for t in tm]
        [tm11, tm12, tm21, tm22] = [t for t in tm]

        nmu = torch.matmul(tm11, mu.T).T + torch.matmul(tm12, ml.T).T
        nml = torch.matmul(tm21, mu.T).T + torch.matmul(tm22, ml.T).T

        mu_prior = torch.cat([nmu, nml], dim=-1)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        cu, cl, cs = covar
        cov_prior = cov_linear_transform(tm, [cu, cl, cs])
    else:
        tm_batched = tm.repeat((mean.shape[0],1))
        mu_prior = tm_batched*mean
        cov_prior = cov_linear_transform(tm_batched, covar, mem=False)

    return mu_prior, cov_prior


def cov_linear_transform(tm: List, covar: List, mem=True) -> List:
    """
    Performs the linear transformation of the covariance matrix. This uses efficient sparse matrix multiplications,
    especially for the covariance matrix. This makes use of the block structure of the covariance matrix and the fact
    that we are only interested in the diagonal elements of the blocks of the resulting covariance matrix.
    :param tm: list of transition matrices
    :param covar: prior covariance
    :param mem: whether to use memory (H=[I,0] observation model) or not
    :return: next prior covariance
    """
    # predict next prior covariance (eq 10 - 12 in paper supplement)
    if mem:
        cu, cl, cs = covar
        [tm11, tm12, tm21, tm22] = [t for t in tm] 

        ncu = torch.matmul(tm11**2, cu.T).T + 2.0 * torch.matmul(tm11 * tm12, cs.T).T + torch.matmul(tm12**2, 
                                                                                                        cl.T).T
        ncl = torch.matmul(tm21**2, cu.T).T + 2.0 * torch.matmul(tm21 * tm22, cs.T).T + torch.matmul(tm22**2,
                                                                                                        cl.T).T
        ncs = torch.matmul(tm21 * tm11, cu.T).T + torch.matmul(tm22 * tm11, cs.T).T + torch.matmul(tm21 * tm12,
                                                                                                cs.T).T + torch.matmul(
            tm22 * tm12, cl.T).T
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        covar_new = tm**2 * covar

        lod = int(covar.shape[-1] / 2)

        ncu = covar_new[:, :lod]
        ncl = covar_new[:, lod:]
        ncs = torch.zeros(ncu.shape[0], ncu.shape[1]).to(device)

    return [ncu,ncl,ncs]


def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0) ##

class Control(nn.Module):
    def __init__(self, action_dim, lsd, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._action_dim = action_dim
        self._lsd = lsd
        self._dtype = dtype
        layers = []
        prev_dim = self._action_dim

        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.LayerNorm(prev_dim)) ### added layer norm
        layers.append(nn.Linear(prev_dim, self._lsd))
        self._control = nn.Sequential(*layers).to(dtype=self._dtype)



    def forward(self, action: torch.Tensor):
        x = self._control(action)

        return x

class ProcessNoise(nn.Module):
    def __init__(self, lsd, init_trans_covar, num_hidden: Iterable[int], activation: str, dtype: torch.dtype = torch.float32):
        super().__init__()

        self._lsd = lsd
        self._dtype = dtype
        init_trans_cov = elup1_inv(init_trans_covar)
        self._log_process_noise = nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd, dtype=self._dtype), init_trans_cov))

    def forward(self):
        x = self._log_process_noise

        return x

class Predict(nn.Module):

    def __init__(self, latent_obs_dim: int, act_dim: int = None, hierarchy_type: str = None, config = None, dtype: torch.dtype = torch.float32):
        """
        TODO: add references to block diagram
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param act_dim: action dimension
        :param hierarchy_type: manager / submanager / worker
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(Predict, self).__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod
        self._action_dim = act_dim

        self._device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        
        self._dtype = dtype
        self._hier_type = hierarchy_type
        ##raise error if hierarchy type is not one of the four
        assert self._hier_type in ["manager", "submanager", "worker", "worker_v2", "ACRKN", "HIPRSSM"], \
                                    "Hierarchy Type should be one of manager, submanager, worker or ACRKN"

        ## get A matrix
        self._A = self.get_transformation_matrix()
        ## convert to nn parameter
        #self._A = [nn.Parameter(t, requires_grad=True) for t in self._A]
        if self._action_dim is not None:
            ### get B matrix for actions
            if self._hier_type in ["worker","ACRKN","HIPRSSM"]:
                ## control neural net
                self._b = Control(self._action_dim, self._lsd, self.c.control_net_hidden_units,
                                    self.c.control_net_hidden_activation).to(self._device)
            else:
                ## get B matrix for abstract action
                self._B = self.get_transformation_matrix()

        if self._hier_type is "submanager" or self._hier_type is "worker" or self._hier_type is "worker_v2" or self._hier_type is "HIPRSSM":
            ## get C matrix for task
            self._C = self.get_transformation_matrix() ##
            ## convert to nn parameter
            #self._C = [nn.Parameter(t, requires_grad=True) for t in self._C]

        
        # These are constant matrices (not learning) used by the model - no gradients required

        self._eye_matrix = torch.eye(self._lod).to(self._device)
        np_mask = np.ones([self._lod, self._lod], dtype=np.float32)
        np_mask = np.triu(np_mask, -self.c.bandwidth) * np.tril(np_mask, self.c.bandwidth)
        self._band_mask = torch.from_numpy(np_mask).to(self._device) 

        # TODO: This is currently a different noise for each dim, not like in original paper (and acrkn)

        self._log_process_noise = ProcessNoise(self._lsd, self.c.trans_covar, self.c.process_noise_hidden_units,
                                                self.c.process_noise_hidden_activation)

    def get_transformation_matrix(self, mem=True) -> None:
        """
        Builds the basis functions for transition model and the nosie
        :return:
        """
        if mem:
            # build state independent basis matrices
            tm_11_full = nn.Parameter(torch.zeros(self._lod, self._lod, dtype=self._dtype).to(self._device))
            tm_12_full = \
                nn.Parameter(0.2 * torch.eye(self._lod, dtype=self._dtype).to(self._device))
            tm_21_full = \
                nn.Parameter(-0.2 * torch.eye(self._lod, dtype=self._dtype).to(self._device))
            tm_22_full = nn.Parameter(torch.zeros(self._lod, self._lod, dtype=self._dtype).to(self._device))
            # apply the mask
            ## eye matrix is added to the diagonal of the transition matrix (TODO: check if this is correct)
            #self._tm_11_full += self._eye_matrix
            #self._tm_22_full += self._eye_matrix
            ##parameter list tm
            tm = nn.ParameterList([tm_11_full, tm_12_full, tm_21_full, tm_22_full])
        else:
            tm = nn.Parameter(torch.rand((self._lsd), dtype=self._dtype, device=self._device)[None, :])
        return tm



    def get_process_noise(self) -> torch.Tensor:
        """
        Compute the process noise covariance matrix
        :return: transition covariance (vector of size lsd)
        """

        process_cov = elup1(self._log_process_noise()).to(self._device)
        # prepare process noise
        lod = int(process_cov.shape[-1] / 2)
        process_cov_upper = process_cov[..., :lod]
        process_cov_lower = process_cov[..., lod:]
        process_cov_side = torch.zeros(process_cov_upper.shape).to(self._device)
        process_cov = [process_cov_upper, process_cov_lower, process_cov_side]
        return process_cov



    #   @torch.jit.script_method
    def forward(self, post_mean_list: Iterable[torch.Tensor], post_cov_list: Iterable[torch.Tensor]) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor]]:
        """
        forward pass through the cell. For proper recurrent model feed back outputs 3 and 4 (next prior belief at next time step)

        :param post_mean_list: list of posterior means at time t that forms the causal factors that are used to predict mean at time t + 1
        :param post_cov_list: list of posterior covariances at time t that forms the causal factors that are used to predict covariance at time t + 1
        :return: prior mean at time t + 1, prior covariance time t + 1
        """
        if self._hier_type == "manager":
            ## Manager
            prior_mean_0, prior_cov_0 = gaussian_linear_transform(self._A, post_mean_list[0], post_cov_list[0])
            if self._action_dim is not None:
                prior_mean_1, prior_cov_1 = gaussian_linear_transform(self._B, post_mean_list[1], post_cov_list[1])
            else:
                prior_mean_1 = torch.zeros_like(post_mean_list[0])
                prior_cov_1 = [torch.zeros_like(post_cov_list[0][0]), torch.zeros_like(post_cov_list[0][1]), torch.zeros_like(post_cov_list[0][2])]
            next_prior_mean = prior_mean_0 + prior_mean_1
            next_prior_cov = prior_cov_0 + prior_cov_1
        elif self._hier_type == "submanager":
            ## Submanager
            prior_mean_0, prior_cov_0 = gaussian_linear_transform(self._A, post_mean_list[0], post_cov_list[0])
            if self._action_dim is not None:
                prior_mean_1, prior_cov_1 = gaussian_linear_transform(self._B, post_mean_list[1], post_cov_list[1])
            else:
                prior_mean_1 = torch.zeros_like(post_mean_list[1])
                prior_cov_1 = [torch.zeros_like(post_cov_list[0][0]), torch.zeros_like(post_cov_list[0][1]), torch.zeros_like(post_cov_list[0][2])]
            prior_mean_2, prior_cov_2 = gaussian_linear_transform(self._C, post_mean_list[-1], post_cov_list[-1])
            next_prior_mean = prior_mean_0 + prior_mean_1 + prior_mean_2
            next_prior_cov = [x + y + z for x, y, z in zip(prior_cov_0, prior_cov_1, prior_cov_2)]
        elif self._hier_type == "worker":
            ## Worker
            prior_mean_0, prior_cov_0 = gaussian_linear_transform(self._A, post_mean_list[0], post_cov_list[0])
            if self._action_dim is not None:
                prior_mean_1 = self._b(post_mean_list[1])
            else:
                prior_mean_1 = torch.zeros_like(post_mean_list[0])
            prior_mean_2, prior_cov_2 = gaussian_linear_transform(self._C, post_mean_list[-1], post_cov_list[-1])

            next_prior_mean = prior_mean_0 + prior_mean_1 + prior_mean_2
            next_prior_cov = [x + z for x, z in zip(prior_cov_0, prior_cov_2)]
        elif self._hier_type == "worker_v2":
            ## Worker
            prior_mean_0, prior_cov_0 = gaussian_linear_transform(self._A, post_mean_list[0], post_cov_list[0])
            if self._action_dim is not None:
                prior_mean_1, prior_cov_1 = gaussian_linear_transform(self._B, post_mean_list[1], post_cov_list[1])
            else:
                prior_mean_1 = torch.zeros_like(post_mean_list[0])
                prior_cov_1 = [torch.zeros_like(post_cov_list[0][0]), torch.zeros_like(post_cov_list[0][1]), torch.zeros_like(post_cov_list[0][2])]
            prior_mean_2, prior_cov_2 = gaussian_linear_transform(self._C, post_mean_list[-1], post_cov_list[-1])

            next_prior_mean = prior_mean_0 + prior_mean_1 + prior_mean_2
            next_prior_cov = [x + y + z for x, y, z in zip(prior_cov_0, prior_cov_1, prior_cov_2)]
        elif self._hier_type == "HIPRSSM":
            ## HIPRSSM
            prior_mean_0, prior_cov_0 = gaussian_linear_transform(self._A, post_mean_list[0], post_cov_list[0])
            if self._action_dim is not None:
                prior_mean_1 = self._b(post_mean_list[1])
            else:
                prior_mean_1 = torch.zeros_like(post_mean_list[0])
            prior_mean_2, prior_cov_2 = gaussian_linear_transform(self._C, post_mean_list[-1], post_cov_list[-1])
            next_prior_mean = prior_mean_0 + prior_mean_1 + prior_mean_2
            next_prior_cov = [x + z for x, z in zip(prior_cov_0, prior_cov_2)]

        elif self._hier_type == "ACRKN":
            ## ACRKN
            prior_mean_0, prior_cov_0 = gaussian_linear_transform(self._A, post_mean_list[0], post_cov_list[0])
            if self._action_dim is not None:
                prior_mean_1 = self._b(post_mean_list[1])
            else:
                prior_mean_1 = torch.zeros_like(post_mean_list[0])
            next_prior_mean = prior_mean_0 + prior_mean_1
            next_prior_cov = prior_cov_0
        ### finally add process noise
        process_covar = self.get_process_noise()
        next_prior_cov = [x+y for x,y in zip(next_prior_cov, process_covar)]

        return next_prior_mean, next_prior_cov








