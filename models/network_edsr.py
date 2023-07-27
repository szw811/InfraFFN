import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, img_range=1.,
                 in_chans=3, res_scale=1,
                 conv=common.default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        # define head module
        m_head = [conv(in_chans, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, in_chans, kernel_size)
        ]

        # self.add_mean = common.MeanShift(img_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        # x = self.add_mean(x)
        x = x / self.img_range + self.mean
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

