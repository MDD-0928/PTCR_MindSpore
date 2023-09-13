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
# WITHOUT WARRANT IES OR CONITTONS OF ANY KINDï¼?either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Data augmentation."""

import math
import random

from ..config import cfg

# from mindspore.dataset.transforms.c_transforms import Compose
# import mindspore.dataset.vision.c_transforms as c_trans
# c_transforms
from mindspore.dataset.transforms import Compose
import mindspore.dataset.vision as vis


"""
class RandomErasing():
    Randomly erases an image patch.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    

    def __init__(
            self,
            probability=0.5,
            sl=0.02,
            sh=0.4,
            r1=0.3,
            mean=None
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for _ in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
"""

def build_train_transforms(
        height,
        width,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []
    if isinstance(transforms, str):
        transforms = [transforms]
    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if transforms:
        transforms = [t.lower() for t in transforms]

    print('Building train transforms ...')
    transform_tr = []

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [vis.Resize((height, width))]

    if 'random_flip' in cfg.TRANSFORM:
        print('+ random flip')
        transform_tr += [vis.RandomHorizontalFlip()]

    if 'pad' in cfg.TRANSFORM:
        print('+ Pad')
        transform_tr += [vis.Pad(10)]

    if 'random crop' in cfg.TRANSFORM:
        print('+ random crop')
        transform_tr += [vis.RandomCrop(size=(height, width))]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std
    normalize = vis.Normalize(mean=[255*i for i in norm_mean], std=[255*i for i in norm_std])
    print('+ Normalization (mean={}, std={})'.format([255*i for i in norm_mean], [255*i for i in norm_std]))
    transform_tr += [normalize]

    print("+ HWC2CHW()")
    transform_tr += [vis.HWC2CHW()]

    if 'random_erase' in cfg.TRANSFORM and cfg.REA:
        print('+ random erase')
        # transform_tr += [RandomErasing(mean=norm_mean)]
        transform_tr += [vis.RandomErasing(value=[int(255*i) for i in norm_mean])]
        
    print('+ to numpy array of range [0, 1]')
    transform_tr += [vis.Rescale(1.0/255.0, 0.0)]
        
    transform_tr = Compose(transform_tr)
    return transform_tr


def build_test_transforms(
        height,
        width,
        norm_mean=None,
        norm_std=None
):
    '''build transforms for test data.'''
    normalize = vis.Normalize(mean=norm_mean, std=norm_std)
    transform_te = Compose([
        vis.Resize((height, width)),
        vis.Rescale(1.0/255.0, 0.0),
        normalize,
        vis.HWC2CHW(),
    ])
    return transform_te
