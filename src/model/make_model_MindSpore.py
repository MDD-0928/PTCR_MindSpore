import mindspore as ms

import mindspore.nn as nn
import mindspore.dataset.vision as vs
import copy
import sys
import os
import numpy as np
from mindspore.common.initializer import initializer, Constant, HeNormal, Normal
from ..model.backbones.PTCR_MindSpore import PTCR
from ..utils.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from ..model.layers.gem_pool_MindSpore import GeneralizedMeanPoolingP as GeM

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
sys.path.append(os.path.dirname(__file__) + os.sep + './')


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = ms.ops.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], axis=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = ms.ops.cat([x, x[:, -2:-1, :]], axis=1)
        x = x.view(batchsize, group, -1, dim)
    # x = torch.transpose(x, 1, 2).contiguous()
    x = ms.ops.swapaxes(x, 1, 2)
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        # nn.init.constant_(m.bias, 0.0)
        print('yeyeyeyeeyeyeyeye')
        m.weight.set_data(initializer(HeNormal(mode='fan_out'), m.weight.shape, m.weight.dtpye))
        m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))

    elif classname.find('Conv') != -1:
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        m.weight.set_data(initializer(HeNormal(mode='fan_in'), m.weight.shape, m.weight.dtpye))
        if m.bias is not None:
            # nn.init.constant_(m.bias, 0.0)
            m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            # nn.init.constant_(m.weight, 1.0)
            # nn.init.constant_(m.bias, 0.0)
            m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))
            m.weight.set_data(initializer(Constant(1.0), m.weight.shape, m.weight.dtype))


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print('ooooooooooo')
        # nn.init.normal_(m.weight, std=0.001)
        m.weight.set_data(initializer(Normal(sigma=0.001), m.weight.shape, m.weight.dtype))
        if m.bias:
            # nn.init.constant_(m.bias, 0.0)
            m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))

class TokenPerception(nn.Cell):
    def __init__(self, num_group, num_classes):
        super().__init__()
        # self.trans = transforms.RandomPerspective(p=1)
        self.trans = vs.RandomPerspective(prob=1.0)

        embed_dims = [64, 128, 256]
        self.num_group = num_group
        self.num_classes = num_classes

        # self.p_GeM = [GeM().cuda() for _ in range(self.num_group)]
        # self.bottalnecks = [nn.BatchNorm1d(embed_dims[2], device="cuda") for _ in range(self.num_group)]
        # self.p_head = [nn.Linear(embed_dims[2], num_classes, device="cuda") for _ in range(self.num_group)]
        self.p_GeM = [GeM() for _ in range(self.num_group)]
        self.bottalnecks = [nn.BatchNorm1d(embed_dims[2]) for _ in range(self.num_group)]
        self.p_head = [nn.Dense(embed_dims[2], num_classes) for _ in range(self.num_group)]

    def construct(self, stage_feats):
        shapes = [feat.shape for feat in stage_feats]
        # scores = torch.zeros(shapes[2][0], self.num_classes).cuda()
        scores = ms.ops.zeros((shapes[2][0], self.num_classes))

        for i in range(self.num_group):
            if i == self.num_group - 1:
                feat = self.p_GeM[i](stage_feats[2][:, :, i * (shapes[2][2] // self.num_group):, :])
            else:
                feat = self.p_GeM[i](stage_feats[2][:, :,
                                     i * (shapes[2][2] // self.num_group):(i + 1) * (shapes[2][2] // self.num_group),
                                     :])
            feat = feat.view(feat.shape[0], -1)
            feat_bn = self.bottalnecks[i](feat)
            scores += self.p_head[i](feat_bn)

        return scores / self.num_group


class build_PTCR(nn.Cell):
    def __init__(self, cfg, num_classes=1000):
        super().__init__()

        embed_dims = [64, 128, 256, 512]
        self.ptcr = PTCR(num_classes=num_classes)
        self.head = nn.Dense(embed_dims[3], num_classes)
        self.bottalneck = nn.BatchNorm1d(embed_dims[3])
        self.TP = cfg.MODEL.TP
        if self.TP:
            self.token_perception = TokenPerception(5, num_classes)

        # if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet21k':
        #     self.load_param(cfg.MODEL.PRETRAIN_PATH)
        #     print('Loading pretrained ImageNet21K model......from {}'.format(cfg.MODEL.PRETRAIN_PATH))

    def construct(self, x, label=None, view_label=None, cam_label=None):
        # print(x.shape)
        feat, stage_feats = self.ptcr(x, label, view_label, cam_label)
        shapes = [feat.shape for feat in stage_feats]
        # print("stage_feats_shape: {}".format(shapes))
        # print("\n")
        # print("feat: {}".format(feat))
        # print("\n")
        # print("feat_type: {}".format(type(feat)))
        # print("\n")
        # print("feat_shape: {}".format(feat.shape))
        # print("\n")
        feat_bn = self.bottalneck(feat)
        # print(feat_bn)
        # print(feat_bn.shape)
        score = self.head(feat_bn)
        

        if self.TP:
            score += self.token_perception(stage_feats)

        if self.training:
            # print(score)
            # print("score: {}".format(score))
            # print("\n")
            # print("score_type: {}".format(type(score)))
            # print("\n")
            # print("score_shape: {}".format(score.shape))
            # print("\n")
            # print("feat: {}".format(feat))
            # print("\n")
            # print("feat_type: {}".format(type(feat)))
            # print("\n")
            # print("feat_shape: {}".format(feat.shape))
            # print("\n")
            return score, feat
        else:
            return feat

    # def load_param(self, trained_path):
    #     param_dict = ms.load_checkpoint(trained_path)
    #     for i in param_dict:
    #         self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
    #     print('Loading pretrained model from {}'.format(trained_path))
    #
    # def load_param_finetune(self, model_path):
    #     param_dict = ms.load_checkpoint(model_path)
    #     for i in param_dict:
    #         self.state_dict()[i].copy_(param_dict[i])
    #     print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_PTCR(cfg, num_class)
    print('===========building PCTR===========')
    return model
