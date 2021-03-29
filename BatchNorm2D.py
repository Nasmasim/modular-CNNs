import torch
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        """
        An implementation of a Batch Normalization over a mini-batch of 2D inputs.

        The mean and standard-deviation are calculated per-dimension over the
        mini-batches and gamma and beta are learnable parameter vectors of
        size num_features.

        Parameters:
        - num_features: C from an expected input of size (N, C, H, W).
        - eps: a value added to the denominator for numerical stability. Default: 1e-5
        - momentum: momentum – the value used for the running_mean and running_var
        computation. Default: 0.1
        - gamma: the learnable weights of shape (num_features).
        - beta: the learnable bias of the module of shape (num_features).
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum   
        
        shape = (1, num_features, 1, 1)    
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        self.moving_avg = torch.zeros(shape)
        self.moving_var = torch.ones(shape)    
    
    # batch normalisation function
    def batch_normalisation(self,X, gamma, beta, moving_avg, moving_var, eps, momentum):
        # check if training
        if not torch.is_grad_enabled():
            X_pred = (X - moving_avg) / torch.sqrt(moving_var + eps)
        else:
            # check shape
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # calculate mean and variance
                avg = X.mean(dim=0)                
                var = ((X - avg) ** 2).mean(dim=0)
            else:
                avg = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - avg) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            
            X_pred = (X - avg) / torch.sqrt(var + eps)
            # Update the mean and variance using moving average
            moving_avg = momentum * moving_avg + (1.0 - momentum) * avg
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        # Scale and shift
        Y = gamma * X_pred + beta  
        return Y, moving_avg.data, moving_var.data

    def forward(self, x):
        """
        During training this layer keeps running estimates of its computed mean and
        variance, which are then used for normalization during evaluation.
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data of shape (N, C, H, W) (same shape as input)
        """
        
        if self.moving_avg.device != x.device:
            self.moving_avg = self.moving_avg.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_avg, self.moving_var = self.batch_norm(
            x, self.gamma, self.beta, self.moving_avg,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y