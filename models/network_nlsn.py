import common_nlsn
import attention
import torch.nn as nn
import torch


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return NLSN(args, dilated.dilated_conv)
    else:
        return NLSN(args)


class NLSN(nn.Module):
    def __init__(self, n_resblocks=32, n_feats=256, scale=4, in_chans=3, img_range=1.0,
                 chunk_size=144, n_hashes=4, res_scale=0.1, conv=common_nlsn.default_conv):
        super(NLSN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        m_head = [conv(in_chans, n_feats, kernel_size)]

        # define body module
        m_body = [attention.NonLocalSparseAttention(
            channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale)]

        for i in range(n_resblocks):
            m_body.append(common_nlsn.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ))
            if (i + 1) % 8 == 0:
                m_body.append(attention.NonLocalSparseAttention(
                    channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale))
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common_nlsn.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, in_chans, kernel_size,
                padding=(kernel_size // 2)
            )
        ]

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
