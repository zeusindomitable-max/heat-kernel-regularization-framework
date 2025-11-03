import torch

class HeatKernel:
    def __init__(self, tau: float):
        self.tau = tau

    def kernel(self, theta, theta_prime):
        d2 = torch.sum((theta - theta_prime) ** 2)
        n = theta.numel()
        return (4 * torch.pi * self.tau) ** (-n / 2) * torch.exp(-d2 / (4 * self.tau))

    def d_tau(self, theta, theta_prime):
        k = self.kernel(theta, theta_prime)
        n = theta.numel()
        d2 = torch.sum((theta - theta_prime) ** 2)
        return k * (-n / (2 * self.tau) + d2 / (4 * self.tau**2))
