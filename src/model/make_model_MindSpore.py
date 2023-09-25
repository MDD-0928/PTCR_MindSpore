import mindspore as ms

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


# def shuffle_unit(features, shift, group, begin=1):
#     batchsize = features.size(0)
#     dim = features.size(-1)
#     # Shift Operation
#     feature_random = ms.ops.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], axis=1)
#     x = feature_random
#     # Patch Shuffle Operation
#     try:
#         x = x.view(batchsize, group, -1, dim)
#     except:
#         x = ms.ops.cat([x, x[:, -2:-1, :]], axis=1)
#         x = x.view(batchsize, group, -1, dim)
#     # x = torch.transpose(x, 1, 2).contiguous()
#     x = ms.ops.swapaxes(x, 1, 2)
#     x = x.view(batchsize, -1, dim)
#
#     return x
#
#
# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
#         # nn.init.constant_(m.bias, 0.0)
#         m.weight.set_data(initializer(HeNormal(mode='fan_out', negative_slope=0), m.weight.shape, m.weight.dtpye))
#         m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
#
#     elif classname.find('Conv') != -1:
#         # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#         m.weight.set_data(initializer(HeNormal(mode='fan_in', negative_slope=0), m.weight.shape, m.weight.dtpye))
#         if m.bias is not None:
#             # nn.init.constant_(m.bias, 0.0)
#             m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
#     elif classname.find('BatchNorm') != -1:
#         if m.affine:
#             # nn.init.constant_(m.weight, 1.0)
#             # nn.init.constant_(m.bias, 0.0)
#             m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
#             m.weight.set_data(initializer(One(), m.weight.shape, m.weight.dtype))
#
#
# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         # nn.init.normal_(m.weight, std=0.001)
#         m.weight.set_data(initializer(Normal(sigma=0.001), m.weight.shape, m.weight.dtype))
#         if m.bias:
#             # nn.init.constant_(m.bias, 0.0)
#             m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))


# class build_transformer_local(nn.Cell):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
#         super(build_transformer_local, self).__init__()
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         self.in_planes = 768
#
#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
#
#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0
#
#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
#                                                         local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
#                                                         stride_size=cfg.MODEL.STRIDE_SIZE,
#                                                         drop_path_rate=cfg.MODEL.DROP_PATH)
#
#         # if pretrain_choice == 'imagenet':
#         #     self.base.load_param(model_path)
#         #     print('Loading pretrained ImageNet model......from {}'.format(model_path))
#
#         block = self.base.blocks[-1]
#         layer_norm = self.base.norm
#         self.b1 = nn.SequentialCell(
#             copy.deepcopy(block),
#             copy.deepcopy(layer_norm)
#         )
#         self.b2 = nn.SequentialCell(
#             copy.deepcopy(block),
#             copy.deepcopy(layer_norm)
#         )
#
#         self.num_classes = num_classes
#         self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         # if self.ID_LOSS_TYPE == 'arcface':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#         #                                              cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = Arcface(self.in_planes, self.num_classes,
#         #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'cosface':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#         #                                              cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = Cosface(self.in_planes, self.num_classes,
#         #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'amsoftmax':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#         #                                              cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#         #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'circle':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
#         #                                              cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
#         #                                  s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#
#         self.classifier = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
#         self.classifier.apply(weights_init_classifier)
#         self.classifier_1 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
#         self.classifier_1.apply(weights_init_classifier)
#         self.classifier_2 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
#         self.classifier_2.apply(weights_init_classifier)
#         self.classifier_3 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
#         self.classifier_3.apply(weights_init_classifier)
#         self.classifier_4 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
#         self.classifier_4.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.beta.requires_grad(False)
#         self.bottleneck.apply(weights_init_kaiming)
#
#         self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_1.bias.requires_grad_(False)
#         self.bottleneck1.beta.requires_grad(False)
#         self.bottleneck_1.apply(weights_init_kaiming)
#
#         self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_2.bias.requires_grad_(False)
#         self.bottleneck_2.beta.requires_grad(False)
#         self.bottleneck_2.apply(weights_init_kaiming)
#
#         self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_3.bias.requires_grad_(False)
#         self.bottleneck_3.beta.requires_grad(False)
#         self.bottleneck_3.apply(weights_init_kaiming)
#
#         self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_4.bias.requires_grad_(False)
#         self.bottleneck_4.beta.requires_grad(False)
#         self.bottleneck_4.apply(weights_init_kaiming)
#
#         self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
#         print('using shuffle_groups size:{}'.format(self.shuffle_groups))
#         self.shift_num = cfg.MODEL.SHIFT_NUM
#         print('using shift_num size:{}'.format(self.shift_num))
#         self.divide_length = cfg.MODEL.DEVIDE_LENGTH
#         print('using divide_length size:{}'.format(self.divide_length))
#         self.rearrange = rearrange
#
#     def construct(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
#
#         features = self.base(x, cam_label=cam_label, view_label=view_label)
#
#         # global branch
#         b1_feat = self.b1(features)  # [64, 129, 768]
#         global_feat = b1_feat[:, 0]  # [64, 768]
#
#         # JPM branch
#         feature_length = features.size(1) - 1
#         patch_length = feature_length // self.divide_length
#         token = features[:, 0:1]
#
#         if self.rearrange:
#             x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
#         else:
#             x = features[:, 1:]
#         # lf_1
#         b1_local_feat = x[:, :patch_length]
#         b1_local_feat = self.b2(ms.ops.cat((token, b1_local_feat), axis=1))
#         local_feat_1 = b1_local_feat[:, 0]  # [64, 768]
#
#         # lf_2
#         b2_local_feat = x[:, patch_length:patch_length * 2]
#         b2_local_feat = self.b2(ms.ops.cat((token, b2_local_feat), axis=1))
#         local_feat_2 = b2_local_feat[:, 0]  # [64, 768]
#
#         # lf_3
#         b3_local_feat = x[:, patch_length * 2:patch_length * 3]
#         b3_local_feat = self.b2(ms.ops.cat((token, b3_local_feat), axis=1))
#         local_feat_3 = b3_local_feat[:, 0]  # [64, 768]
#
#         # lf_4
#         b4_local_feat = x[:, patch_length * 3:patch_length * 4]
#         b4_local_feat = self.b2(ms.ops.cat((token, b4_local_feat), axis=1))
#         local_feat_4 = b4_local_feat[:, 0]  # [64, 768]
#
#         feat = self.bottleneck(global_feat)  # [64, 768]
#
#         local_feat_1_bn = self.bottleneck_1(local_feat_1)  # [64, 768]
#         local_feat_2_bn = self.bottleneck_2(local_feat_2)  # [64, 768]
#         local_feat_3_bn = self.bottleneck_3(local_feat_3)  # [64, 768]
#         local_feat_4_bn = self.bottleneck_4(local_feat_4)  # [64, 768]
#
#         if self.training:
#             if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
#                 cls_score = self.classifier(feat, label)
#             else:
#                 cls_score = self.classifier(feat)  # [64, class]
#                 cls_score_1 = self.classifier_1(local_feat_1_bn)  # [64, class]
#                 cls_score_2 = self.classifier_2(local_feat_2_bn)  # [64, class]
#                 cls_score_3 = self.classifier_3(local_feat_3_bn)  # [64, class]
#                 cls_score_4 = self.classifier_4(local_feat_4_bn)  # [64, class]
#             return [cls_score, cls_score_1, cls_score_2, cls_score_3,
#                     cls_score_4
#                     ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
#                         local_feat_4]  # global feature for triplet utils
#         else:
#             if self.neck_feat == 'after':
#                 return ms.ops.cat(
#                     [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], axis=1)
#             else:
#                 return ms.ops.cat(
#                     [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], axis=1)
#
#     # def load_param(self, trained_path):
#     #     param_dict = ms.load_checkpoint(trained_path)
#     #     for i in param_dict:
#     #         self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#     #     print('Loading pretrained model from {}'.format(trained_path))
#     #
#     # def load_param_finetune(self, model_path):
#     #     param_dict = ms.load_checkpoint(model_path)
#     #     for i in param_dict:
#     #         self.state_dict()[i].copy_(param_dict[i])
#     #     print('Loading pretrained model for finetuning from {}'.format(model_path))


class TokenPerception(nn.Cell):
    def __init__(self, num_group, num_classes):
        super().__init__()
        # self.trans = transforms.RandomPerspective(p=1)
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
