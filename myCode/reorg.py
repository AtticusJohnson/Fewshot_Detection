import torch
import torch.nn as nn


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 3)
        C = x.data.size(0)
        H = x.data.size(1)
        W = x.data.size(2)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(C, int(H/hs), hs, int(W/ws),
                   ws).transpose(2, 3).contiguous()
        x = x.view(C, int(H/hs*W/ws), int(hs*ws)
                   ).transpose(1, 2).contiguous()
        x = x.view(C, int(hs*ws), int(H/hs), int(W/ws)
                   ).transpose(0, 1).contiguous()
        x = x.view(int(hs*ws*C), int(H/hs), int(W/ws))
        return x


x = torch.linspace(1, 16, 16).reshape(4, 4).unsqueeze(0)
print(f'prev x is: \n{x}')

reg = Reorg()
x = reg(x)
for i in range(x.shape[0]):
    print(f'after x[{i}] is: \n{x[i, ...]}')
