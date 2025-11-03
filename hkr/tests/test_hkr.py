from hkr.kernel import HeatKernel
import torch

def test_heatkernel():
    hk = HeatKernel(tau=0.5)
    a = torch.randn(3)
    b = torch.randn(3)
    val = hk.kernel(a, b)
    assert val > 0
  
