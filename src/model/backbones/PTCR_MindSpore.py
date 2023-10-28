# import torch
# import torch.nn.functional as F
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from itertools import repeat
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Constant, TruncatedNormal, Normal, Zero, One
from ..layers.gem_pool_MindSpore import GeneralizedMeanPoolingP as GeM
from functools import partial
import math
import numpy as np
from mindspore import jit_class

class DropPath(nn.Cell):
    def __init__(self, drop_prob, ndim=1):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(p=float(drop_prob))
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = ms.Tensor(ms.ops.ones(shape), dtype=ms.dtype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ms.ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode='valid'),
            act_layer(),
            nn.BatchNorm2d(hidden_features),
        )

        self.proj = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                              pad_mode='pad', group=hidden_features, has_bias=True)
        self.proj_act = act_layer()
        self.proj_bn = nn.BatchNorm2d(hidden_features)

        
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode='valid'),
            nn.BatchNorm2d(out_features),
        )
        
        self.drop = nn.Dropout(p=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), m.weight.shape, m.weight.dtype))
            if isinstance(m, nn.Dense) and m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
        elif isinstance(m, nn.BatchNorm2d):
            m.beta.set_data(initializer(Zero(), m.beta.shape, m.beta.dtype))
            m.gamma.set_data(initializer(One(), m.gamma.shape, m.gamma.dtype))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.group
            m.weight.set_data(initializer(Normal(sigma=math.sqrt(2.0 / fan_out)), m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, has_bias=True, pad_mode='valid')
            self.norm = nn.LayerNorm((dim,))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), m.weight.shape, m.weight.dtype))
            if isinstance(m, nn.Dense) and m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
        elif isinstance(m, nn.LayerNorm):
            m.beta.set_data(initializer(Zero(), m.beta.shape, m.beta.dtype))
            m.gamma.set_data(initializer(One(), m.gamma.shape, m.gamma.dtype))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.group
            m.weight.set_data(initializer(Normal(sigma=math.sqrt(2.0 / fan_out)), m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), m.weight.shape, m.weight.dtype))
            if isinstance(m, nn.Dense) and m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
        elif isinstance(m, nn.LayerNorm):
            m.beta.set_data(initializer(Zero(), m.beta.shape, m.beta.dtype))
            m.gamma.set_data(initializer(One(), m.gamma.shape, m.gamma.dtype))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.group
            m.weight.set_data(initializer(Normal(sigma=math.sqrt(2.0 / fan_out)), m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class IBN(nn.Cell):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def construct(self, x):
        split = ms.ops.split(x, self.half, axis=1)
        out1 = self.IN(split[0])
        out2 = self.BN(split[1])
        out = ms.ops.cat((out1, out2), axis=1)
        return out


class BlurPool(nn.Cell):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = ms.Tensor(a[:, None] * a[None, :], dtype=ms.dtype.float32)
        filt = filt / ms.ops.sum(filt)
        self.filt = ms.Parameter(filt[None, None, :, :].tile((self.channels, 1, 1, 1)).astype(ms.dtype.float32), name='filt',
                                 requires_grad=False)
        

        self.pad = get_pad_layer(pad_type)(tuple(self.pad_sizes))

    def construct(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return ms.ops.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class ConvPatch(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=[384, 192], patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        filt_size = patch_size
        patch_size = tuple(repeat(patch_size, 2))

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=1,
                      padding=patch_size[0] // 2, pad_mode='pad', has_bias=False),
            IBN(embed_dim),
            nn.ReLU(),
            BlurPool(embed_dim, filt_size=filt_size, stride=stride)
        )
        self.norm = nn.LayerNorm((embed_dim,))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):   
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), m.weight.shape, m.weight.dtype))
            if isinstance(m, nn.Dense) and m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
        elif isinstance(m, nn.LayerNorm):
            m.beta.set_data(initializer(Zero(), m.beta.shape, m.beta.dtype))
            m.gamma.set_data(initializer(One(), m.gamma.shape, m.gamma.dtype))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.group
            m.weight.set_data(initializer(Normal(sigma=math.sqrt(2.0 / fan_out)), m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(start_dim=2).swapaxes(1, 2)
        x = self.norm(x)

        return x, H, W

@jit_class
class AuxiliaryEmbedding():
    def __init__(self, B, D, cam_label, view_label):
        super().__init__()
        self.B = B
        self.D = D
        self.out = self.get_encode(cam_label, view_label)
        
    def __call__(self):
        return self.out

    def get_encode(self, *items):
        arg = [item for item in items if item is not None]
        aux_embed = ms.ops.zeros(self.B)
        for C, T in enumerate(arg):
            aux_embed += ms.ops.sin((C + 1) / 10000 ** (2 * T / self.D))
        return aux_embed


class PTCR(nn.Cell):
    def __init__(self, img_size=[384, 192], in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 depths=[4, 8, 36, 4], sr_ratios=[4, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = ms.ops.linspace(0, drop_path_rate, sum(depths))
        cur = 0

        for i in range(num_stages):
            patch_embed = ConvPatch(
                img_size=img_size if i == 0 else [img_size[0] // (2 ** (i + 1)), img_size[1] // (2 ** (i + 1))],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])

            block = nn.CellList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=nn.LayerNorm,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer((embed_dims[i],))
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.global_pool = GeM()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), m.weight.shape, m.weight.dtype))
            if isinstance(m, nn.Dense) and m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
        elif isinstance(m, nn.LayerNorm):
            m.beta.set_data(initializer(Zero(), m.beta.shape, m.beta.dtype))
            m.gamma.set_data(initializer(One(), m.gamma.shape, m.gamma.dtype))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.group
            m.weight.set_data(initializer(Normal(sigma=math.sqrt(2.0 / fan_out)), m.weight.shape, m.weight.dtype))
            if m.bias is not None:
                m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))



    def construct(self, x, label=None, cam_label=None, view_label=None ):
        B = x.shape[0]
        maps = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            if i == 0:
                D = x.shape[2]
                # print(1231232)
                aux_embed = AuxiliaryEmbedding(B, D, cam_label, view_label)
                for j in range(B):
                    x[j] += 0.55 * aux_embed.out[j]
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if i != self.num_stages - 1:
                maps.append(x)
            else:
                x = self.global_pool(x)
                x = x.view(x.shape[0], -1)
        
        return x, maps
