import torch
import torch.nn as nn
from parameters import param

device = param.device
class LaplacianOp(nn.Module):
    def __init__(self):
        super(LaplacianOp, self).__init__()
        self.conv_kernel = nn.Parameter(torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float64, device=device),
                                        requires_grad=False)


    def forward(self, inputs, dx=1.0, dy=1.0, boundary=None):
        '''
        :param inputs: [batch, iH, iW], torch.float
        :return: laplacian of inputs
        '''
        unsqueezed = False
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 0)
            unsqueezed = True
        if boundary is None:
            inputs1 = torch.cat([inputs[:, -1:, :], inputs, inputs[:, :1, :]], dim=1)
            inputs2 = torch.cat([inputs1[:, :, -1:], inputs1, inputs1[:, :, :1]], dim=2)
        else:
            T0 = torch.ones_like(inputs[:, -1:, :])*boundary
            inputs1 = torch.cat([T0, inputs, T0], dim=1)
            T0 = torch.ones_like(inputs1[:, :, -1:])*boundary
            inputs2 = torch.cat([T0, inputs1, T0], dim=2)
        conv_inputs = torch.unsqueeze(inputs2, dim=1)
        result = torch.nn.functional.conv2d(input=conv_inputs, weight=self.conv_kernel).squeeze(dim=1) / (dx*dy)
        if unsqueezed:
            result = torch.squeeze(result, 0)
        '''
        # dimension of result? same as inputs argument?
        '''
        return result