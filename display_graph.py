import torch
import torch.nn as nn
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func


class MLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def dynamo_backend(gm: torch.fx.GraphModule, example_inputs):
    """Receives the FX graph captured by TorchDynamo (torch-level ops)."""
    print("=" * 30, "Dynamo FX graph", "=" * 30)
    gm.graph.print_tabular()
    print("\n--- generated code ---")
    print(gm.code)
    return gm.forward  # run eagerly


def aten_compiler(gm: torch.fx.GraphModule, example_inputs):
    print("=" * 30, "AOTAutograd (ATen) FX graph", "=" * 30)
    gm.graph.print_tabular()
    print("\n--- generated code ---")
    print(gm.code)
    return make_boxed_func(gm.forward)


aten_backend = aot_autograd(fw_compiler=aten_compiler)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    
    sym_gm = torch.fx.symbolic_trace(model)
    print("=" * 30, "Symbolic FX graph", "=" * 30)
    sym_gm.graph.print_tabular()
    print("\n\n")
    
    x = torch.randn(4, 16, device=device)

    # 1. Graph captured by Dynamo (high-level torch ops / nn.Module calls)
    compiled = torch.compile(model, backend=dynamo_backend)
    compiled(x)

    # 2. Lowered ATen-level graph (what Inductor receives)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=aten_backend)
    with torch.no_grad():
        compiled(x)


if __name__ == "__main__":
    main()
