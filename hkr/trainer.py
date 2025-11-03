import torch
from .kernel import HeatKernel
from .curvature import CurvatureRegularizer

class HKRTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.curv = CurvatureRegularizer()

    def step(self, x, y):
        pred = self.model(x)
        loss_task = torch.nn.functional.mse_loss(pred, y)
        grads = torch.autograd.grad(loss_task, self.model.parameters(), create_graph=True)
        ricci_term = self.curv.ricci_proxy(grads)
        h_trace = self.curv.hutchinson_trace(loss_task, next(self.model.parameters()))
        total = loss_task + self.cfg['alpha']*ricci_term + self.cfg['gamma']*h_trace
        total.backward()
        return total.item()
