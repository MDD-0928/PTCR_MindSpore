# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""utils.py"""
import mindspore.numpy as msnp
import mindspore as ms
import mindspore.ops as P
from mindspore import nn
from ..config import cfg


class CrossEntropyLoss(nn.Cell):
    r"""Cross entropy utils with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the utils function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, eps=0.1, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps if label_smooth else 0
        self.logsoftmax = nn.LogSoftmax(axis=-1)
        self.zeros = P.Zeros()
        self.expand_dims = P.ExpandDims()

    def construct(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        # print(log_probs)
        # print(log_probs.shape)
        # print(type(log_probs))
        depth = log_probs.shape[0]
        onehot = nn.OneHot(depth=depth, axis=-1)
        targets = onehot(targets)
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        return (-targets * log_probs).mean(0).sum()


class MarginRankingLoss(nn.Cell):
    """
    class of MarginRankingLoss
    """

    def __init__(self, margin=0.):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ge = P.GreaterEqual()
        self.sum = P.ReduceSum(keep_dims=True)
        self.mean = P.ReduceMean(keep_dims=True)

    def construct(self, input1, input2, y):
        """
        MarginRankingLoss
        """
        temp1 = - self.sub(input1, input2)
        temp2 = self.mul(temp1, y)
        temp3 = self.add(temp2, self.margin)
        temp3_mask = self.ge(temp3, 0)

        loss = self.mean(temp3 * temp3_mask)
        return loss

def normalize(x, axis=-1):

    x = 1. * x / (ms.Tensor.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.shape) == 2
    assert dist_mat.shape[0] == dist_mat.shape[1]
    N = dist_mat.shape[0]

    # shape [N, N]
    temp = ms.ops.broadcast_to(labels, (N, N))
    is_pos = temp.equal(temp.t())
    is_neg = temp.ne(temp.t())
    # is_pos = labels.expand(N, N).equal(labels.expand(N, N).t())
    # is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = ms.ops.max(
        dist_mat[is_pos].view(N, -1), 1, keepdims=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = ms.ops.min(
        dist_mat[is_neg].view(N, -1), 1, keepdims=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    '''if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(ms.ops.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = ms.ops.gather_elements(
            ind[is_pos].view(N, -1), 1, relative_p_inds.data)
        n_inds = ms.ops.gather_elements(
            ind[is_neg].view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds
    '''
    
    return dist_ap, dist_an

class TripletLoss(nn.Cell):

    def __init__(self, margin=None, hard_factor=0.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
            
    def construct(self, global_feat, labels, normalize_feature=False):
        
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = pdist_ms(global_feat, global_feat)
        
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        # y = dist_an.new().resize_as_(dist_an).fill_(1)
        y = dist_an.new_ones(dist_an.shape)
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
            
        return loss
        
        
class OriTripletLoss(nn.Cell):
    """Triplet utils with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, batch_size=64):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = MarginRankingLoss(self.margin)

        self.pow_ms = P.Pow()
        self.sum = P.ReduceSum(keep_dims=True)
        self.transpose = P.Transpose()
        self.mul = P.Mul()
        self.add = P.Add()
        self.sub = P.Sub()
        self.sqrt = P.Sqrt()
        self.equal = P.Equal()
        self.notequal = P.NotEqual()
        self.cat = P.Concat()
        self.ones_like = P.OnesLike()
        self.squeeze = P.Squeeze()
        self.unsqueeze = P.ExpandDims()
        self.max = P.ReduceMax(keep_dims=True)
        self.min = P.ReduceMin(keep_dims=True)
        self.cat = P.Concat()
        # self.matmul = P.MatMul()
        self.expand = P.BroadcastTo((batch_size, batch_size))
        self.cast = P.Cast()

    def construct(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        # Compute pairwise distance, replace by the official when merged
        dist = self.pow_ms(inputs, 2)
        dist = self.sum(dist, 1)
        dist = self.expand(dist)
        dist = self.add(dist, self.transpose(dist, (1, 0)))

        temp1 = P.matmul(inputs, self.transpose(inputs, (1, 0)))
        temp1 = self.mul(-2, temp1)
        dist = self.add(dist, temp1)
        # for numerical stability, clip_value_max=? why must set?
        dist = P.composite.clip_by_value(
            dist, clip_value_min=1e-12, clip_value_max=100000000)
        dist = self.sqrt(dist)

        # For each anchor, find the hardest positive and negative
        targets = self.expand(targets)
        mask_pos = self.cast(self.equal(
            targets, self.transpose(targets, (1, 0))), ms.int8)
        mask_neg = self.cast(self.notequal(
            targets, self.transpose(targets, (1, 0))), ms.int8)
        dist_ap = self.max(dist * mask_pos, 1).squeeze()
        dist_an = self.min(self.max(dist * mask_neg, 1)
                           * mask_pos + dist, 1).squeeze()

        # Compute ranking hinge utils
        y = self.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss[0]


class TripletLossWRT(nn.Cell):
    """
    class of WRT TripletLoss
    """

    def __init__(self):
        super(TripletLossWRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def construct(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        dist_mat = pdist_ms(inputs, inputs)
        dist_mat_n = dist_mat.shape[0]
        bs = P.BroadcastTo((dist_mat_n, dist_mat_n))
        equal = P.Equal()
        ne = P.NotEqual()
        cast = P.Cast()
        # matmul = P.MatMul()
        op = P.ReduceSum()
        is_pos = cast(equal(bs(targets), bs(targets).T), ms.float32)
        is_neg = cast(ne(bs(targets), bs(targets).T), ms.float32)
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = op(dist_ap * weights_ap, 1)
        closest_negative = op(dist_an * weights_an, 1)

        y = msnp.full(furthest_positive.shape, 1, dtype=ms.float32)

        loss = self.ranking_loss(closest_negative - furthest_positive, y)
        return loss


def pdist_ms(emb1, emb2):
    """
    pdist mindspore
    """
    m, n = emb1.shape[0], emb2.shape[0]
    pow_ms = P.Pow()
    bc1 = P.BroadcastTo((m, n))
    bc2 = P.BroadcastTo((n, m))
    sqrt = P.Sqrt()
    # matmul = P.MatMul()
    emb1_pow = bc1(pow_ms(emb1, 2).sum(axis=1, keepdims=True))
    emb2_pow = bc2(pow_ms(emb2, 2).sum(axis=1, keepdims=True)).T

    dist_mtx = emb1_pow + emb2_pow
    # a = Tensor(1, dtype=ms.float32)
    # b = Tensor(-2, dtype=ms.float32)
    dist_mtx = addmm(dist_mtx, 1, -2, emb1, emb2.T)
    dist_mtx = P.composite.clip_by_value(
        dist_mtx, clip_value_min=1e-12, clip_value_max=1e7)
    output = sqrt(dist_mtx)
    return output


def addmm(dist, a, b, m1, m2):
    """
    addmm mindspore
    """
    y1 = a * dist
    y2 = P.matmul(m1, m2)
    y2 = b * y2
    y = y1 + y2
    return y


def softmax_weights(dist, mask):
    """
    softmax_weights
    """
    argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
    # matmul = P.MatMul()
    op = P.ReduceSum(keep_dims=True)
    exp = P.Exp()
    _, max_v = argmax(dist * mask)
    diff = dist - max_v
    tmp_z = op(exp(diff) * mask, 1)
    tmp_z = tmp_z + 1e-6
    tmp_w = exp(diff) * mask / tmp_z
    return tmp_w
    
    


class PTCRLoss(nn.Cell):
    '''
    wrapped Loss, passed to Model
    '''

    def __init__(self, ce=None, tri=None):
        super().__init__()
        self.ce = ce
        self.tri = tri
        self.loss_weight = cfg.MODEL.ID_LOSS_WEIGHT 
        self.loss_weight2 = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        

    def construct(self, logits, labels):
        '''
        forward
        '''
        score, feat = logits
       
        target = labels
        
        if isinstance(score, list):
            acc = (score[0].max(axis=1, return_indices=True)[1] == target).float().mean()
        else:
            acc = (score.max(axis=1, return_indices=True)[1] == target).float().mean()
            
        print(acc)

        if isinstance(score, list):
            ID_LOSS = [ms.ops.cross_entropy(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.5 * ID_LOSS + 0.5 * ms.ops.cross_entropy(score[0], target)
        else:
            ID_LOSS = ms.ops.cross_entropy(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [self.tri(feats, target)[0] for feats in feat[1:]]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * self.tri(feat[0], target)[0]
        else:
            TRI_LOSS = self.tri(feat, target)
                
        return self.loss_weight * ID_LOSS + \
            self.loss_weight2 * TRI_LOSS
