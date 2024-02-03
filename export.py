import numpy as np
from mindspore import Tensor, Model, context, export, load_checkpoint, load_param_into_net
from src.model.make_model_export import make_model
from src.config.configs import get_config
from src.dataset.dataset_2 import dataset_creator
import argparse
import os
cfg = get_config()
context.set_context(mode=context.GRAPH_MODE,
                        device_target='Ascend')

context.set_context(device_id=cfg.MODEL.DEVICE_ID)

dataset1_num_classes1, dataset1, camera_num, view_num = dataset_creator(
        root=cfg.DATASETS.ROOT_DIR, height=cfg.INPUT.HEIGHT,
        width=cfg.INPUT.WIDTH, dataset=cfg.DATASETS.NAMES,
        norm_mean=cfg.INPUT.PIXEL_MEAN, norm_std=cfg.INPUT.PIXEL_STD,
        batch_size_train=cfg.SOLVER.IMS_PER_BATCH, workers=cfg.DATALOADER.NUM_WORKERS,
        cuhk03_labeled=cfg.DATALOADER.cuhk03_labeled, cuhk03_classic_split=cfg.DATALOADER.cuhk03_classic_split
        , mode='train')

input_tensor1 = Tensor(np.ones([32, 3, 384, 128]).astype(np.float32))
input_tensor2 = Tensor(np.ones([32]).astype(np.int32))

PTCR = make_model(cfg, num_class=dataset1_num_classes1, camera_num=camera_num, view_num=view_num)
export(PTCR, input_tensor1, input_tensor2, file_name='PTCR', file_format='MINDIR')
print('export PTCR.mindir successfully')
