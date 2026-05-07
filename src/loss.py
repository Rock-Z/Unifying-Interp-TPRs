from torch import nn

class WeightedMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        weights = 1.0 / (y_true**2 + self.eps)
        return (weights * (y_pred - y_true)**2).mean()
