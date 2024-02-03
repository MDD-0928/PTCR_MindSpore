import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import os
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import argparse
from logger import setup_logger
from src.model.make_model_export import make_model
from src.dataset.dataset_2 import dataset_creator
from src.config.configs import get_config
from src.metric import distance, rank
cfg=get_config()
logger = setup_logger('PTCR', cfg.OUTPUT_DIR, if_train=False)
logger.info('PTCR_Pyramidal Transformer with Conv-Patchify for Person Re-identification')
logger.info("Using model in the path :{}".format(cfg.TEST.WEIGHT))
logger.info("Testing with config:\n{}".format(cfg))
set_seed(cfg.SOLVER.SEED)

class CustomWithEvalCell(nn.Cell):
    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._network = network

    def construct(self, data):
        outputs = self._network(data)
        return outputs


def eval_net(net=None):
    '''prepare to eval net'''
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='Ascend')
    context.set_context(device_id=cfg.MODEL.DEVICE_ID)

    num_train_classes, query_dataset, camera_num, view_num = dataset_creator(
        root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
        width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
        norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
        batch_size_test=cfg.TEST.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
        cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split
        , mode='query')
    num_train_classes, gallery_dataset, camera_num, view_num = dataset_creator(
        root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
        width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
        norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
        batch_size_test=cfg.SOLVER.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
        cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split
        , mode='gallery')
    
    net = make_model(cfg, num_class=num_train_classes, camera_num=camera_num, view_num=view_num)
    
    param_dict = load_checkpoint(cfg.TEST.WEIGHT)
    
    del_list = []

    for i in param_dict.keys():
        if ('head' in i):
            del_list.append(i)

    for i in del_list:
        param_dict.pop(i)
    
    a, b=load_param_into_net(net, param_dict)
    
    print('load over')

    do_eval(net, query_dataset, gallery_dataset)


def do_eval(net, query_dataset, gallery_dataset):
    '''eval the net, called in EvalCallback'''

    net.set_train(False)
    net_eval = CustomWithEvalCell(net)

    def feature_extraction(eval_dataset):
        f_, pids_, camids_ = [], [], []
        for data in eval_dataset.create_dict_iterator():
            imgs, pids, camids = data['img'], data['pid'], data['camid']
            features = net_eval(imgs)
            f_.append(features)
            pids_.extend(pids.asnumpy())
            camids_.extend(camids.asnumpy())
        concat = ops.Concat(axis=0)
        f_ = concat(f_)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return f_, pids_, camids_

    logger.info('Extracting features from query set ...')    
    print('Extracting features from query set ...')
    qf, q_pids, q_camids = feature_extraction(query_dataset)
    logger.info('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    logger.info('Extracting features from gallery set ...')
    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = feature_extraction(gallery_dataset)
    logger.info('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    if cfg.TEST.FEAT_NORM == 'yes':
        l2_normalize = ops.L2Normalize(axis=1)
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)

    logger.info('Computing distance matrix with metric=euclidean...')
    print('Computing distance matrix with metric=euclidean...')

    distmat = distance.compute_distance_matrix(qf, gf, 'euclidean')
    distmat = distmat.asnumpy()

    if not cfg.DATALOADER.use_metric_cuhk03:
        logger.info('Computing CMC mAP mINP ...')
        print('Computing CMC mAP mINP ...')
        cmc, mean_ap, mean_inp = rank.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=cfg.DATALOADER.use_metric_cuhk03
        )
    else:
        print('Computing CMC and mAP ...')
        cmc, mean_ap = rank.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=cfg.DATALOADER.use_metric_cuhk03
        )
    logger.info('** Results **')
    logger.info('ckpt={}'.format(cfg.TEST.WEIGHT))
    logger.info('mAP: {:.2%}'.format(mean_ap))
    logger.info('mINP: {:.2%}'.format(mean_inp))
    logger.info('CMC curve')
    print('** Results **')
    print('ckpt={}'.format(cfg.TEST.WEIGHT))
    print('mAP: {:.2%}'.format(mean_ap))
    print('mINP: {:.2%}'.format(mean_inp))
    print('CMC curve')
    ranks = [1, 5, 10, 20]
    i = 0
    for r in ranks:
        logger.info('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
        i += 1

if __name__ == '__main__':

    eval_net()
