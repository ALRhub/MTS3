import torch
from typing import Tuple

nn = torch.nn


def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)


class Encoder(nn.Module):

    def __init__(self, lod: int, hidden_units_list: list, output_normalization: str = "post", activation='softplus'):
        """Gaussian Encoder, as described in ICLR Paper (if output_normalization=post)
        :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
        :param output_normalization: when to normalize the output:
            - post: after output layer (as described in ICML paper)
            - pre: after last hidden layer, that seems to work as well in most cases but is a bit more principled
            - none: (or any other string) not at all

        """
        super(Encoder, self).__init__()
        self._hidden_units_list = hidden_units_list
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                                "torch.nn.ModuleList or else the hidden weights are " \
                                                                "not found by the optimizer"
        self._ln_pre = nn.LayerNorm(size_last_hidden)
        self._ln_post = nn.LayerNorm(lod)
        self._mean_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._log_var_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._softplus = nn.Softplus()
        self._output_normalization = output_normalization
        self._activation = activation
        
    def _build_hidden_layers(self):
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        layers = []
        last_hidden = self._inp_shape
        # hidden layers 
        for hidden_dim in self._hidden_units_list:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = obs

        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            #h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)
            h = self._ln_pre(h)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            #mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)
            mean = self._ln_post(mean)
        elif self._output_normalization.lower() == "none":
            mean = mean

        log_var = self._log_var_layer(h)
        if self._activation == 'softplus':
            var = self._softplus(log_var) + 0.0001
        else:
            var = elup1(log_var)
        return mean, var

class EncoderSimple(nn.Module):

    def __init__(self, lod: int, output_normalization: str = "post", activation='softplus'):
        """Gaussian Encoder, as described in ICML Paper (if output_normalization=post)
        :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
        :param output_normalization: when to normalize the output:
            - post: after output layer (as described in ICML paper)
            - pre: after last hidden layer, that seems to work as well in most cases but is a bit more principled
            - none: (or any other string) not at all

        """
        super(EncoderSimple, self).__init__()
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                                "torch.nn.ModuleList or else the hidden weights are " \
                                                                "not found by the optimizer"
        self._mean_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)

        self._output_normalization = output_normalization
        self._activation = activation

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = obs
        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)

        return mean

class ConvEncoder(nn.Module):
    def __init__(self, lod: int, output_normalization: str = "post", activation='softplus'):
        super().__init__()
        """Gaussian Convolutional Encoder, as described in ICLR Paper (if output_normalization=post)
                :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
                :param output_normalization: when to normalize the output:
                    - post: after output layer (as described in ICML paper)
                    - pre: after last hidden layer, that seems to work as well in most cases but is a bit more principled
                    - none: (or any other string) not at all

                """
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                               "torch.nn.ModuleList or else the hidden weights are " \
                                                               "not found by the optimizer"
        self._mean_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._log_var_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._softplus = nn.Softplus()

        self._output_normalization = output_normalization
        self._activation = activation


    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for conv encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.unsqueeze(obs,1)

        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)

        log_var = self._log_var_layer(h)
        if self._activation == 'softplus':
            var = self._softplus(log_var) + 0.0001
        else:
            var = elup1(log_var)
        return mean, var


