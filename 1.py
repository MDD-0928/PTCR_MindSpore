import mindspore
import numpy as np
from mindspore import Tensor
shape_tensor1 = Tensor(np.zeros((4, 3), dtype=np.float32))
data=[[1,2,3],[2,3,4],[3,4,5],[1,2,3]]

a=Tensor(data)
print(a.shape)
b=mindspore.ops.pow(a,2)
c=b.sum(axis=1, keepdims=True)
print(b)
print(c)
op_pow = mindspore.ops.Pow()
mat1 = op_pow(a, 2).sum(
    axis=1, keepdims=True).expand_as(shape_tensor1)
print(mat1)