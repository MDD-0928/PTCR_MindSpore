import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import os 
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import argparse
from src.model.make_model_MindSpore import make_model
from src.dataset.dataset import dataset_creator
from src.config import cfg
from src.metric import distance, rank
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
    #if cfg.device_target == "Ascend":
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
    
    net = make_model(cfg, num_class=num_train_classes, camera_num=camera_num, view_num=view_num)

    param_dict = load_checkpoint("/data1/mqx_log/PTCR_GRAPH_3/debug_logs/checkpoint/market1501/mdd-120_736.ckpt")

    load_param_into_net(net, param_dict)

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

    print('Extracting features from query set ...')
    qf, q_pids, q_camids = feature_extraction(query_dataset)
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))
    
    
    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = feature_extraction(gallery_dataset)
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))
    


    if cfg.TEST.FEAT_NORM=='yes':
        l2_normalize = ops.L2Normalize(axis=1)
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)

    print('Computing distance matrix with metric=euclidean...')

    distmat = distance.compute_distance_matrix(qf, gf, 'euclidean')
    distmat = distmat.asnumpy()


    if not cfg.DATALOADER.use_metric_cuhk03:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./src/PTCR.yml", help="path to config file", type=str
    )

    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
        
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    eval_net()
