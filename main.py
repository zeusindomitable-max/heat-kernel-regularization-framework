import torch
from hkr.trainer import HKRTrainer
from hkr.utils import load_config
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    cfg = load_config()
    model = SimpleModel()
    trainer = HKRTrainer(model, cfg)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    for epoch in range(cfg['epochs']):
        loss = trainer.step(x, y)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss:.5f}")
