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
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND�?either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Data augmentation."""

import math
import random
import mindspore
from mindspore.dataset.vision import Inter
from ..config import cfg
from mindspore.dataset.transforms.transforms import PyTensorOperation
# from mindspore.dataset.transforms.c_transforms import Compose
# import mindspore.dataset.vision.c_transforms as c_trans
# c_transforms
from mindspore.dataset.transforms import Compose
from mindspore import dtype as mstype
import mindspore.dataset.vision as vis
from mindspore import Tensor

"""
def _get_pixels(per_pixel, rand_color, patch_size, dtype=mstype.float32):

    if per_pixel:
        return mindspore.ops.normal(patch_size, 0, 1, dtype=mstype.float32)
    elif rand_color:
        return mindspore.ops.normal((patch_size[0], 1, 1), 0, 1, dtype=mstype.float32)
    else:
        return mindspore.ops.zeros((patch_size[0], 1, 1), dtype=mstype.float32)


class RandomErasing(PyTensorOperation):

    def __init__(self, probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3, max_aspect=None, mode='const',
                 min_count=1, max_count=None, num_splits=0):
        super().__init__()
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def _execute_py(self, np_img):

        if len(np_img.shape) == 3:
            self._erase(np_img, *np_img.shape, np_img.dtype)
        else:
            batch_size, chan, img_h, img_w = np_img.shape
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(np_img[i], chan, img_h, img_w, np_img.dtype)
        return input

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype)
                    break
"""
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
    transform_tr += [vis.Resize((height, width), interpolation=Inter.BICUBIC)]

    if 'random_flip' in cfg.TRANSFORM:
        print('+ random flip')
        transform_tr += [vis.RandomHorizontalFlip()]

    if 'pad' in cfg.TRANSFORM:
        print('+ Pad')
        transform_tr += [vis.Pad(10)]

    if 'random crop' in cfg.TRANSFORM:
        print('+ random crop')
        transform_tr += [vis.RandomCrop(size=(height, width))]

    print('+ to numpy array of range [0, 1]')
    transform_tr += [vis.ToTensor()]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std
    normalize = vis.Normalize(mean=norm_mean, std=norm_std, is_hwc=False)
    print('+ Normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in cfg.TRANSFORM and cfg.REA:
        print('+ random erase')
        transform_tr += [vis.RandomErasing(value='random', max_attempts=10)]

    transform_tr = Compose(transform_tr)
    return transform_tr


def build_test_transforms(
        height,
        width,
        norm_mean=None,
        norm_std=None
):
    '''build transforms for test data.'''
    normalize = vis.Normalize(mean=norm_mean, std=norm_std, is_hwc=False)
    transform_te = Compose([
        vis.Resize((height, width)),
        vis.ToTensor(),
        normalize,
        #vis.HWC2CHW(),
    ])
    return transform_te
