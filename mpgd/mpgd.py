import torch

class mpgd():
    def __init__(self, threshold=0.005, toploss):
        self.threshold = threshold #threshold in mpgd
        self.loss = toploss #loss functions in pytorch

    def mpgd_loss(self, y, y_hat):
        diff     = torch.abs(y-y_hat)/torch.max(y_hat)
        err_mask = torch.zeros_like(y)
        err_mask[diff>self.threshold]=1.0
        count = torch.sum(error_mask)
        if count>0:
            loss = self.loss(y[diff>self.threshold], y_hat[diff>self.threshold])
        else:
            loss = self.loss(y, y_hat)

    
    def __call__(self, y, y_hat):
        return self.mpgd_loss(y, y_hat)