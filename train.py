#coding=UTF-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND锟?either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================
'''train net'''

import warnings
import os
import os.path as osp
import numpy as np
import argparse
import mindspore as ms
import mindspore.nn as nn

from mindspore.common import set_seed
from mindspore import Tensor, Model, context
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import (Callback, ModelCheckpoint, CheckpointConfig, LossMonitor,
                                      TimeMonitor, SummaryCollector)
from mindspore.communication.management import init
from test import do_eval
from src.utils.loss import OriTripletLoss, CrossEntropyLoss, TripletLoss, PTCRLoss
from src.model.make_model_MindSpore import make_model
from src.utils.local_adapter import get_device_id, get_device_num
from src.config import cfg
from src.dataset.dataset import dataset_creator
from src.utils.lr_generator import step_lr, multi_step_lr, warmup_step_lr

import mindcv.optim as optim_ms
import mindcv.scheduler as scheduler_ms
from mindspore.train import Accuracy

set_seed(cfg.SOLVER.SEED)

class LossCallBack(LossMonitor):
    """
    Monitor the utils in training.
    If the utils in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        '''check utils at the end of each step.'''
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid utils, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, utils is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                       cur_step_in_epoch, loss), flush=True)


class EvalCallBack(Callback):
    '''
    Train-time Evaluation
    '''

    def __init__(self, net, eval_per_epoch):
        self.net = net
        self.eval_per_epoch = eval_per_epoch

        _, self.query_dataset, _, _ = dataset_creator(
            root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
            width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
            norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
            batch_size_test=cfg.TEST.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
            cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split,
            mode='query')
        _, self.gallery_dataset, _, _ = dataset_creator(
            root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
            width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
            norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
            batch_size_test=cfg.TEST.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
            cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split,
            mode='gallery')

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            do_eval(self.net, self.query_dataset, self.gallery_dataset)

def check_isfile(fpath):
    '''check whether the path is a file.'''
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


# def load_from_checkpoint(net):
#     '''load parameters when resuming from a checkpoint for training.'''
#     param_dict = load_checkpoint(cfg.TEST.WEIGHT)
#     if param_dict:
#         if param_dict.get("epoch_num") and param_dict.get("step_num"):
#             cfg.start_epoch = int(param_dict["epoch_num"].data.asnumpy())
#             cfg.start_step = int(param_dict["step_num"].data.asnumpy())
#         else:
#             cfg.start_epoch = 0
#             cfg.start_step = 0
#         load_param_into_net(net, param_dict)
#     else:
#         raise ValueError("Checkpoint file:{} is none.".format(
#             cfg.checkpoint_file_path))


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(
        cfg.OUTPUT_DIR, cfg.CHECKPOINT_PATH, cfg.DATASETS.NAMES)

    return ckpt_save_dir


def get_callbacks(num_batches):
    '''get all callback list'''
    time_cb = TimeMonitor(data_size=num_batches)
    loss_cb = LossCallBack()
    summary_collector = SummaryCollector(
        summary_dir='/data1/mqx_log/PTCR_MindSpore/summary_dir', collect_freq=cfg.SOLVER.CHECKPOINT_PERIOD * num_batches)

    cb = [time_cb, loss_cb, summary_collector]
    
    # print(num_batches)

    ckpt_append_info = [
        {"epoch_num": cfg.SOLVER.START_EPOCH, "step_num": cfg.SOLVER.START_EPOCH}]
    # cfg_ck = CheckpointConfig(save_checkpoint_steps=cfg.SOLVER.CHECKPOINT_PERIOD * num_batches,
                            #   keep_checkpoint_max=5, append_info=ckpt_append_info)
    cfg_ck = CheckpointConfig(save_checkpoint_steps = cfg.SOLVER.CHECKPOINT_PERIOD * num_batches,
                              keep_checkpoint_max=30, append_info=ckpt_append_info)

    ckpt_cb = ModelCheckpoint(
        prefix="mdd", directory=set_save_ckpt_dir(), config=cfg_ck)
    cb += [ckpt_cb]

    return cb


def train_net():
    """train net"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='GPU')
    # device_num = get_device_num()
    # if cfg.MODEL.DEVICE == "CPU":
    # device_id = get_device_id()
    # context.set_context(device_id=device_id)
    #     if device_num > 1:
    #         context.reset_auto_parallel_context()
    #         context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel',
    #                                           gradients_mean=True)
    #         init()


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    num_classes, dataset1, camera_num, view_num = dataset_creator(
        root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
        width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
        norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
        batch_size_train=cfg.SOLVER.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
        cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split
        , mode='train')

    num_batches = dataset1.get_dataset_size()
   
    net = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    param_dict = load_checkpoint("mdd-120_736.ckpt")
    load_param_into_net(PTCR, param_dict)
  
    ptcrloss = PTCRLoss(
        ce=CrossEntropyLoss(num_classes=num_classes,
                            label_smooth=cfg.MODEL.LABELSMOOTH),
        tri=TripletLoss(margin=cfg.SOLVER.MARGIN)
        )

    params = []
    for param in net.get_parameters():
        if not param.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in param.name:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in param.name or "arcface" in param.name:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [param], "lr": lr, "weight_decay": weight_decay}]
                              
    lr_sche = scheduler_ms.create_scheduler(
        steps_per_epoch=num_batches,
        scheduler="cosine_decay",
        lr=cfg.SOLVER.BASE_LR,
        min_lr=0.002 * cfg.SOLVER.BASE_LR,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_factor=0.01,
        decay_epochs=cfg.SOLVER.MAX_EPOCHS - cfg.SOLVER.WARMUP_EPOCHS,
        num_epochs=cfg.SOLVER.MAX_EPOCHS,
        num_cycles=1,
        cycle_decay=1.0,
        lr_epoch_stair=True
    )

    opt3 = optim_ms.create_optimizer(net.trainable_params(), opt='adamw', lr=1e-7, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    model2 = Model(network=net, optimizer=opt3, loss_fn=ptcrloss)

    callbacks = get_callbacks(num_batches)

    model2.train(cfg.SOLVER.MAX_EPOCHS, dataset1, callbacks, dataset_sink_mode=False)

    print("======== Train success ========")


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

    train_net()
