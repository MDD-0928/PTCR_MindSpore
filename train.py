# coding=UTF-8
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
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND??either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================
'''train net'''

import warnings
import os
import time
import argparse
import os.path as osp
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import Tensor, Model, context
from mindspore import load_checkpoint
from mindspore.train.callback import (Callback, ModelCheckpoint, CheckpointConfig, LossMonitor,
                                      TimeMonitor, SummaryCollector)
from mindspore.communication.management import init
from mindspore import _checkparam as Validator
from mindspore.train import Accuracy
from test import do_eval
from src.utils.loss import OriTripletLoss, CrossEntropyLoss, TripletLoss, PTCRLoss
from src.model.make_model_MindSpore import make_model, load_paramdict_into_net
from src.config.configs import get_config
from src.dataset.dataset_2 import dataset_creator
from logger import setup_logger
import mindcv.optim as optim_ms
import mindcv.scheduler as scheduler_ms
cfg = get_config()
logger = setup_logger('PTCR', cfg.OUTPUT_DIR, if_train=True)
logger.info('PTCR_Pyramidal Transformer with Conv-Patchify for Person Re-identification')
logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
logger.info("Running with config:\n{}".format(cfg))
set_seed(cfg.SOLVER.SEED)

class TimeCallBack(TimeMonitor):
    def __init__(self, data_size=0):
        super(TimeCallBack, self).__init__()
        self.datasize = data_size

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.

        Args:
           run_context (RunContext): Context of the process running. For more details,
                   please refer to :class:`mindspore.train.RunContext`.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.datasize
        cb_params = run_context.original_args()
        mode = cb_params.get("mode", "")
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num
        Validator.check_positive_int(step_size)

        step_seconds = epoch_seconds / step_size
        logger.info("{} epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format
              (mode.title(), epoch_seconds, step_seconds))
        print("{} epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format
              (mode.title(), epoch_seconds, step_seconds), flush=True)
              
              
class LossCallBack(LossMonitor):
    """
    Monitor the utils in training.
    If the utils in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0, batch_size=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch
        self.size = batch_size
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
        if self._per_print_times != 0 and (cb_params.cur_step_num-(cb_params.cur_epoch_num + int(self.has_trained_epoch)*self.size)) % 100 == 0:
            logger.info("epoch: %s step: %s, utils is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                       cur_step_in_epoch, loss))
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


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(
        cfg.OUTPUT_DIR, cfg.CHECKPOINT_PATH, cfg.DATASETS.NAMES)

    return ckpt_save_dir


def get_callbacks(num_batches):
    '''get all callback list'''
    time_cb = TimeCallBack(data_size=num_batches)
    loss_cb = LossCallBack(batch_size=num_batches)

    cb = [time_cb, loss_cb]

    ckpt_append_info = [
        {"epoch_num": cfg.SOLVER.START_EPOCH, "step_num": cfg.SOLVER.START_EPOCH}]
   
    cfg_ck = CheckpointConfig(save_checkpoint_steps=cfg.SOLVER.CHECKPOINT_PERIOD * num_batches,
                              keep_checkpoint_max=30, append_info=ckpt_append_info)

    ckpt_cb = ModelCheckpoint(
        prefix="PTCR", directory=set_save_ckpt_dir(), config=cfg_ck)
    cb += [ckpt_cb]

    return cb


def train_net():
    """train net"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target='Ascend')

    context.set_context(device_id=cfg.MODEL.DEVICE_ID)

    num_classes, dataset1, camera_num, view_num = dataset_creator(
        root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
        width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
        norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
        batch_size_train=cfg.SOLVER.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
        cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split, mode='train')

    num_batches = dataset1.get_dataset_size()

    net = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    param_dict = load_checkpoint(cfg.SOLVER.PRETRAIN_WEIGHT)

    ptcrloss = PTCRLoss(
        ce=CrossEntropyLoss(num_classes=num_classes,
                            label_smooth=cfg.MODEL.LABELSMOOTH),
        tri=TripletLoss(margin=cfg.SOLVER.MARGIN)
    )
    base_lr = float(cfg.SOLVER.BASE_LR)

    lr_sche = scheduler_ms.create_scheduler(
        steps_per_epoch=num_batches,
        scheduler="cosine_decay",
        lr=base_lr,
        min_lr=0.5*base_lr,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        decay_epochs=cfg.SOLVER.MAX_EPOCHS - cfg.SOLVER.WARMUP_EPOCHS,
        num_epochs=cfg.SOLVER.MAX_EPOCHS,
        num_cycles=1,
        cycle_decay=1.0,
        lr_epoch_stair=True
    )

    net = load_paramdict_into_net(net, param_dict)
    
    opt3 = optim_ms.create_optimizer(net.trainable_params(), opt='adamw', lr=lr_sche,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    model2 = Model(network=net, optimizer=opt3, loss_fn=ptcrloss, amp_level='O3')

    callbacks = get_callbacks(num_batches)

    model2.train(cfg.SOLVER.MAX_EPOCHS, dataset1, callbacks, dataset_sink_mode=False)

    print("======== Train success ========")


if __name__ == '__main__':

    train_net()
