import torch
import numpy as np
from utils.TimeDistributed import TimeDistributed
from agent.worldModels.SensorEncoders.propEncoder import EncoderSimple
from agent.worldModels.Decoders.propDecoder import SimpleDecoder
from typing import Tuple
optim = torch.optim
nn = torch.nn


class RNNBaseline(nn.Module):
    def __init__(self, input_shape=None, action_dim=None, config=None, use_cuda_if_available: bool = True):
        """
        TODO: Gradient Clipping?
        :param input_shape: shape of the input
        :param action_dim: dimension of the action space
        :param config: Config Dict
        :param use_cuda_if_available: use cuda if available
        """
        super(RNNBaseline, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        if config == None:
            raise ValueError("config cannot be None, pass an omegaConf File")
        else:
            self.c = config

        self._obs_shape = input_shape
        self._action_dim = action_dim
        self._lod = self.c.rnn.latent_obs_dim
        self._lsd = 2 * self._lod

        # parameters
        self._enc_out_normalization = self.c.rnn.enc_out_norm

        # main model
        obs_enc = EncoderSimple(self._obs_shape[-1], self._lod, self.c.rnn.obs_encoder)
        act_enc = EncoderSimple(self._action_dim, self._lod, self.c.rnn.act_encoder)
        enc = EncoderSimple(2*self._lod, self._lsd, self.c.rnn.encoder)
        self._obs_enc = obs_enc.to(self._device)
        self._act_enc = act_enc.to(self._device)
        self._enc = enc.to(self._device)

        if self.c.rnn.type.lower() == 'gru':
            self._lstm_layer = nn.GRU(input_size= 2 * self._lod, hidden_size=5 * self._lod, batch_first=True).to(self._device)
        else:
            self._lstm_layer = nn.LSTM(input_size=2 * self._lod, hidden_size=5 * self._lod, batch_first=True).to(self._device)

        obsDec = SimpleDecoder(latent_state_dim = 5* self._lod, out_dim = self._obs_shape[-1], config = self.c.rnn.obs_decoder)
        self._dec = obsDec.to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_dec_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError


    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, obs_valid_batch: torch.Tensor) -> Tuple[float, float]:
        """Forward Pass oF RNN Baseline
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :return: mean and variance
        """
        # here masked values are set to zero. You can also put an unrealistic value like a negative number.
        obs_masked_batch = obs_batch * obs_valid_batch
        w_obs = self._obs_enc(obs_masked_batch)
        w_obs = w_obs
        act_obs = self._act_enc(act_batch)
        input_batch = torch.cat([w_obs,act_obs], dim=-1)
        w = self._enc(input_batch)
        z, y = self._lstm_layer(w)
        out_mean, out_var = self._dec(z)
        return out_mean, out_var