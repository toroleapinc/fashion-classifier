"""Data augmentation utilities."""
import torch
import numpy as np

class Cutout:
    """Randomly mask out a square patch."""
    def __init__(self, length=8):
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(y + self.length // 2, 0, h)
        x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0
        return img * torch.from_numpy(mask).unsqueeze(0)

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam
