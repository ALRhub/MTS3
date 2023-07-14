import torch

nn = torch.nn


class TimeDistributed(nn.Module):

    def __init__(self, module, low_mem=False, num_outputs=1):
        """
        Makes a torch model time distributed. If the original model works with Tensors of size [batch_size] + data_shape
        this wrapper makes it work with Tensors of size [batch_size, sequence_length] + data_shape
        :param module: The module to wrap
        :param low_mem: Default is to the fast but high memory version. If you run out of memory set this to True
                        (it will be slower than)
            - low memory version: simple forloop over the time axis -> slower but consumes less memory
            - not low memory version: "reshape" and then process all at once -> faster but consumes more memory
        :param num_outputs: Number of outputs of the original module (really the number of outputs,
               not the dimensionality, e.g., for the normal RKN encoder that should be 2 (mean and variance))
        """

        super(TimeDistributed, self).__init__()
        self._module = module
        if num_outputs > 1:
            self.forward = self._forward_low_mem_multiple_outputs if low_mem else self._forward_multiple_outputs
        else:
            self.forward = self._forward_low_mem if low_mem else self._forward
        self._num_outputs = num_outputs

    def _forward(self, *args):
        input_shapes = [args[i].shape for i in range(len(args))]
        batch_size, seq_length = input_shapes[0][0], input_shapes[0][1]
        out = self._module(*[x.contiguous().view(batch_size * seq_length, *input_shapes[i][2:]) for i, x in enumerate(args)])
        return out.view(batch_size, seq_length, *out.shape[1:])

    def _forward_multiple_outputs(self, *args):
        input_shapes = [args[i].shape for i in range(len(args))]
        batch_size, seq_length = input_shapes[0][0], input_shapes[0][1]
        outs = self._module(*[x.reshape(batch_size * seq_length, *input_shapes[i][2:]) for i, x in enumerate(args)])
        out_shapes = [outs[i].shape for i in range(self._num_outputs)]

        return [outs[i].reshape(batch_size, seq_length, *out_shapes[i][1:]) for i in range(self._num_outputs)]

    def _forward_low_mem(self, x):
        out = []
        unbound_x = x.unbind(1)
        for x in unbound_x:
            out.append(self._module(x))
        return torch.stack(out, dim=1)

    def _forward_low_mem_multiple_outputs(self, x):
        out = [[] for _ in range(self._num_outputs)]
        unbound_x = x.unbind(1)
        for x in unbound_x:
            outs = self._module(x)
            [out[i].append(outs[i]) for i in range(self._num_outputs)]
        return [torch.stack(out[i], dim=1) for i in range(self._num_outputs)]