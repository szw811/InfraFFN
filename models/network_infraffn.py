import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_partition2(x, window_size):
    """
    Args:
        x: (B, C, H, W)  pytorch的卷积默认tensor格式为(B, C, H, W)
        window_size (tuple[int]): window size(M)
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    # view: -> [B, C, H//Wh, Wh, W//Ww, Ww]
    x = x.view(B, C, H // window_size[0], window_size[1], W // window_size[0], window_size[1])
    # permute: -> [B, H//Wh, W//Ww, Wh, Ww, C]
    # view: -> [B*num_windows, Wh, Ww, C]
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    num_windows = H//Wh * W//Ww
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Wh, Ww, C] -> [B, H//Wh, W//Ww, Wh, Ww, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H//Wh, Wh, W//Ww, Ww, C]
    # view: [B, H//Wh, Wh, W//Ww, Ww, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def window_reverse2(windows, window_size, H: int, W: int):
    """ Windows reverse to feature map.
    [B * H // win * W // win , win*win , C] --> [B, C, H, W]
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # view: [B*num_windows, N, C] -> [B, H//window_size, W//window_size, window_size, window_size, C]
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # permute: [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, C, H//Wh, Wh, W//Ww, Ww]
    # view: [B, C, H//Wh, Wh, W//Ww, Ww] -> [B, C, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixAttention(nn.Module):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention_block (W-MSA) module
    with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention_block heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention_block weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, dwconv_kernel_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim

        self.window_size = window_size
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        relative_coords = self._get_rel_pos()
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj_attn = nn.Linear(dim, dim)
        self.proj_attn_norm = nn.LayerNorm(dim)

        self.proj_cnn_1 = nn.Linear(dim, dim)
        self.proj_cnn_norm_1 = nn.LayerNorm(dim)

        self.proj_cnn_2 = nn.Linear(dim, dim)
        self.proj_cnn_norm_2 = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # conv branch
        self.dwconv5x5 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=5,
                padding=5 // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1),

        )


        self.projection_1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.projection_2 = nn.Conv2d(dim, dim // 2, kernel_size=1)

        self.conv_norm_1 = nn.BatchNorm2d(dim // 2)
        self.conv_norm_2 = nn.BatchNorm2d(dim // 2)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1)
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.conv_norm = nn.LayerNorm(dim)
        # final projection
        self.projection_cat = nn.Linear(dim * 2, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_rel_pos(self):
        """
            Get pair-wise relative position index for each token inside the window.
            Args:
                window_size (tuple[int]): window size
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """

        x_atten = self.proj_attn_norm(self.proj_attn(x))

        x_cnn_1 = self.proj_cnn_norm_1(self.proj_cnn_1(x))

        x_cnn_1 = window_reverse2(x_cnn_1, self.window_size, H, W)

        x_cnn_2 = self.proj_cnn_norm_2(self.proj_cnn_2(x))

        x_cnn_2 = window_reverse2(x_cnn_2, self.window_size, H, W)

        # conv branch

        x_cnn_1 = self.dwconv3x3(x_cnn_1)

        x_cnn_1 = self.projection_1(x_cnn_1)

        x_cnn_2 = self.dwconv5x5(x_cnn_2)
        x_cnn_2 = self.projection_2(x_cnn_2)

        x_cnn_1 = self.conv_norm_1(x_cnn_1)
        x_cnn_2 = self.conv_norm_2(x_cnn_2)

        x_cnn = torch.cat([x_cnn_1, x_cnn_2], dim=1)
        channel_interaction = self.channel_interaction(x_cnn)

        B_, N, C = x_atten.shape

        qkv = self.qkv(x_atten).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # channel interaction

        x_cnn2v = torch.sigmoid(channel_interaction).reshape(-1, 1, self.num_heads, 1, C // self.num_heads)
        v = v.reshape(x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads)
        v = v * x_cnn2v
        v = v.reshape(-1, self.num_heads, N, C // self.num_heads)

        q = q * self.scale  # Q/sqrt(dk)
        attn = (q @ k.transpose(-2, -1))  # Q*K^{T} / sqrt(dk)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # num_windows

            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_atten = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # spatial interaction
        x_spatial = window_reverse2(x_atten, self.window_size, H, W)

        spatial_interaction = self.spatial_interaction(x_spatial)

        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn

        x_cnn = window_partition2(x_cnn, self.window_size)
        x_cnn = self.conv_norm(x_cnn)

        # concat
        x_atten = self.attn_norm(x_atten)

        x = torch.cat([x_cnn, x_atten], dim=2)
        x = self.projection_cat(x)

        # proj: -> [num_windows*B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixBlock(nn.Module):
    r""" Mixing Block in MixFormer.
    Modified from Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention_block heads.
        window_size (int): Window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        shift_size (int): Shift size for SW-MSA.
            We do not use shift in MixFormer. Default: 0
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=8, dwconv_kernel_size=3, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = MixAttention(
            dim, window_size=(self.window_size, self.window_size), dwconv_kernel_size=dwconv_kernel_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape

        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        # [B, L, C] --- L: H * W
        shortcut = x
        x = self.norm1(x)

        # reshape(): -> [B, H, W, C]
        x = x.reshape(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            # [B, Hp, Wp, C]
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, Hp, Wp, mask=attn_mask)


        # view(): -> [num_windows*B, window_size, window_size, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # window_reverse(): -> [B, Hp, Wp, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            # [B, H', W', C] -> [B, Hp, Wp, C]
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            # [B, Hp, Wp, C]
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # view(): -> [B, H*W, C]
        x = x.view(B, H * W, C)

        # FFN
        # [B, H*W, C]
        x = shortcut + self.drop_path(x)
        # mlp: -> [B, H*W, C]
        # +: -> [B, H*W, C]
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic layer for one stage in MixFormer.
    Modified from Swin Transformer BasicLayer.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention_block heads.
        window_size (int): Local window size.
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end
            of the layer. Default: None
        out_dim (int): Output channels for the downsample layer. Default: 0.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, dwconv_kernel_size=3, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            MixBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                     dwconv_kernel_size=dwconv_kernel_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)])

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, H, W):
        x_blk = x
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x_blk = blk(x_blk, None)
        x_conv = self.conv(x_blk.transpose(1, 2).reshape(-1, self.dim, blk.H, blk.W))
        x = x + x_conv.reshape(-1, self.dim, blk.H * blk.W).transpose(1, 2)

        return H, W, x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class InfraFFN(nn.Module):
    """ A PaddlePaddle impl of MixFormer:
    MixFormer: Mixing Features across Windows and Dimensions (CVPR 2022, Oral)
    Modified from Swin Transformer.
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
            Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention_block heads in different layers.
        window_size (int): Window size. Default: 7
        dwconv_kernel_size (int): kernel size for depth-wise convolution.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the
            patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding.
            Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False
    """

    def __init__(self, img_size=24, patch_size=1, in_chans=3, embed_dim=192, img_range=1.,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=8, dwconv_kernel_size=3,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=2, upsampler='pixelshuffle', resi_connection='1conv',
                 use_checkpoint=False, **kwargs):
        super(InfraFFN, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                dwconv_kernel_size=dwconv_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):

        _, _, Wh, Ww = x.shape

        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            H, W, x = layer(x, Wh, Ww)

        x = self.norm(x)
        return x

    def forward(self, x):
        """
        :param x: input feature (B, C, H, W)
        :return: x : [B, C, H, W]
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        H_1, W_1 = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x_conv_first = x
            x = self.forward_features(x)
            x = x.transpose(1, 2).reshape(-1, self.embed_dim, H_1, W_1)
            # x = self.conv_after_body(x) + x # 残差对象有问题
            x = self.conv_after_body(x) + x_conv_first
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.forward_features(x)
            x = x.transpose(1, 2).reshape(-1, self.embed_dim, H_1, W_1)
            x = self.conv_after_body(x) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.forward_features(x)
            x = x.transpose(1, 2).reshape(-1, self.embed_dim, H, W)
            x = self.conv_after_body(x) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            x_first = self.forward_features(x_first)
            x_first = x_first.transpose(1, 2).reshape(-1, self.embed_dim, H, W)
            res = self.conv_after_body(x_first) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]
