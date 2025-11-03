import torch

class CurvatureRegularizer:
    def ricci_proxy(self, grad_params):
        return sum(torch.sum(g ** 2) for g in grad_params)

    def hutchinson_trace(self, loss_fn, params, n_samples=3):
        traces = []
        for _ in range(n_samples):
            v = torch.randint(0, 2, params.shape, dtype=params.dtype, device=params.device) * 2 - 1
            grad1 = torch.autograd.grad(loss_fn, params, create_graph=True)[0]
            hv = torch.autograd.grad(grad1, params, grad_outputs=v, retain_graph=True)[0]
            traces.append((v * hv).sum())
        return torch.mean(torch.stack(traces))
      
