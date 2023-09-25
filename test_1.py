import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import os
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import argparse
from src.model.make_model_MindSpore_1 import make_model
from src.dataset.dataset_1 import dataset_creator
from src.config import cfg
from src.metric import distance, rank_1
from src.utils.local_adapter import get_device_id

set_seed(1234)


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
                        device_target='GPU')
    # if cfg.device_target == "Ascend":
    #   device_id = get_device_id()
    #   context.set_context(device_id=device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

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
        batch_size_test=cfg.TEST.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
        cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split
        , mode='gallery')

    # num_train_classes, dataset1, camera_num, view_num = dataset_creator(
    # root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
    # width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
    # norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
    # batch_size_train=cfg.SOLVER.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
    # cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split
    # , mode='train')

    for i in dataset1.create_dict_iterator():
        print(i)
        print(" ")
        print(" ")
    # for data in dataset1.create_dict_iterator():
    # print(1)
    # print(dataset1.get_col_names())

    # if net is None:
    #   net = make_model(num_train_classes)
    #  param_dict = load_checkpoint(
    #     cfg.checkpoint_file_path, filter_prefix='epoch_num')
    # params_not_loaded = load_param_into_net(net, param_dict)
    # print(params_not_loaded)
    net = make_model(cfg, num_class=num_train_classes, camera_num=camera_num, view_num=view_num)
    # print(net.parameters_dict())
    # print(query_dataset)
    # print(gallery_dataset)
    param_dict = load_checkpoint("/data1/mqx_log/PTCR_GRAPH_1/debug_logs/checkpoint/market1501/mdd-61_736.ckpt")

    load_param_into_net(net, param_dict)
    print('load over')

    # do_eval(net, query_dataset, dataset1)
    do_eval(net, dataset1, query_dataset, gallery_dataset)

def do_eval(net, dataset1, query_dataset, gallery_dataset):
    # def do_eval(net, query_dataset, dataset1):
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

    def feature_extraction_1(eval_dataset):
        f_, pids_ = [], []
        for data in eval_dataset.create_dict_iterator():
            imgs, pids = data['img'], data['pid']
            features = net_eval(imgs)
            f_.append(features)
            pids_.extend(pids.asnumpy())

        concat = ops.Concat(axis=0)
        f_ = concat(f_)
        pids_ = np.asarray(pids_)

        return f_, pids_

    print('Extracting features from query set ...')
    qf, q_pids, q_camids = feature_extraction(query_dataset)
    # qf, q_pids, q_camids = feature_extraction(dataset1)
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    print('Extracting features from gallery set ...')
    gf, g_pids= feature_extraction_1(dataset1)
    # qf, q_pids, q_camids = feature_extraction(dataset1)
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    # print("qf")
    # print(qf)
    # print("gf")
    # print(gf)

    # if cfg.TEST.FEAT_NORM=='yes':
    # l2_normalize = ops.L2Normalize(axis=1)
    # qf = l2_normalize(qf)
    # gf = l2_normalize(gf)

    print('Computing distance matrix with metric=euclidean...')
    distmat = distance.compute_distance_matrix(qf, gf, 'euclidean')
    distmat = distmat.asnumpy()

    if not cfg.DATALOADER.use_metric_cuhk03:
        print('Computing CMC mAP mINP ...')
        cmc, mean_ap, mean_inp = rank_1.evaluate_rank(
            distmat,
            q_pids,
            g_pids,

            use_metric_cuhk03=cfg.DATALOADER.use_metric_cuhk03
        )
    else:
        print('Computing CMC and mAP ...')
        cmc, mean_ap = rank_1.evaluate_rank(
            distmat,
            q_pids,
            g_pids,

            use_metric_cuhk03=cfg.DATALOADER.use_metric_cuhk03
        )

    print('** Results **')
    print('ckpt={}'.format(cfg.OUTPUT_DIR))
    print('mAP: {:.2%}'.format(mean_ap))
    print('mINP: {:.2%}'.format(mean_inp))
    print('CMC curve')
    ranks = [1, 5, 10, 20]
    i = 0
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
        i += 1

    # net.set_train(True)
    # print(net.training)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./src/PTCR.yml", help="path to config file", type=str
    )

    # parser.add_argument("opts", help="Modify config options using the command-line", default=None,
    #                     nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    eval_net()
