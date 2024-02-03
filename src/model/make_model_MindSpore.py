import mindspore as ms
from mindspore import load_param_into_net
import mindspore.nn as nn
import mindspore.dataset.vision as vs
import copy
import sys
import os
from mindspore.common.initializer import initializer, Constant, HeNormal, Normal, Zero, One
from ..model.backbones.PTCR_MindSpore import PTCR
from ..utils.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from ..model.layers.gem_pool_MindSpore import GeneralizedMeanPoolingP as GeM

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
sys.path.append(os.path.dirname(__file__) + os.sep + './')

class TokenPerception(nn.Cell):
    def __init__(self, num_group, num_classes):
        super().__init__()

        self.trans = vs.RandomPerspective(prob=1.0)

        embed_dims = [64, 128, 256]
        self.num_group = num_group
        self.num_classes = num_classes

        self.p_GeM = nn.CellList()
        for _ in range(self.num_group):
            self.p_GeM.append(GeM())
        self.bottalnecks = nn.CellList()
        for _ in range(self.num_group):
            self.bottalnecks.append(nn.BatchNorm1d(embed_dims[2]))
        self.p_head = nn.CellList()
        for _ in range(self.num_group):
            self.p_head.append(nn.Dense(embed_dims[2], num_classes))


    def construct(self, stage_feats):
        shapes = [feat.shape for feat in stage_feats]

        scores = ms.ops.zeros((shapes[2][0], self.num_classes))

        for i in range(self.num_group):
            if i == self.num_group - 1:
                feat = self.p_GeM[i](stage_feats[2][:, :, i * (shapes[2][2] // self.num_group):, :])
            else:
                feat = self.p_GeM[i](stage_feats[2][:, :,
                                     i * (shapes[2][2] // self.num_group):(i + 1) * (shapes[2][2] // self.num_group), :])
            feat = feat.view(feat.shape[0], -1)
            feat_bn = self.bottalnecks[i](feat)
            scores += self.p_head[i](feat_bn)

        return scores / self.num_group

def load_paramdict_into_net(net, param_dict):

    del_list = []
    for param in net.ptcr.get_parameters():
        param.requires_grad = False
    for i in param_dict.keys():
        if ('head' in i):
            del_list.append(i)
    for i in del_list:
        param_dict.pop(i)
    load_param_into_net(net, param_dict)
    return net

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
        

    def construct(self, x, label=None, view_label=None, cam_label=None):
        
        feat, stage_feats = self.ptcr(x, label, view_label, cam_label)
        
        shapes = [feat.shape for feat in stage_feats]
        
        feat_bn = self.bottalneck(feat)
        score = self.head(feat_bn)

        if self.TP:
            score += self.token_perception(stage_feats)

        if self.training:
            
            return score, feat
        else:
            return feat

def make_model(cfg, num_class, camera_num, view_num):
    model = build_PTCR(cfg, num_class)
    print('===========building PCTR===========')
    return model
