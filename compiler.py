import torch
import torch.fx as fx
from torch._decomp import get_decompositions
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func

from fusion_pointwise import fuse_pointwise


def compiler(gm: fx.GraphModule, example_inputs):
    gm = fuse_pointwise(gm)
    return make_boxed_func(gm.forward)


my_backend = aot_autograd(
    fw_compiler=compiler,
    bw_compiler=compiler,
    decompositions=get_decompositions([
        torch.ops.aten.gelu, 
        torch.ops.aten.silu
    ]),
)