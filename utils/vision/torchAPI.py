import torch
nn = torch.nn

class Reshape(nn.Module):
    """Standard module that reshapes/views a tensor"""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)