import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        """
        An implementation of a max-pooling layer.

        Parameters:
        - kernel_size: the size of the window to take a max over
        """
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """

        W = x.shape[3]
        output_size = int((W-self.kernel_size)/self.kernel_size)+1
        unfold_input = F.unfold(x,self.kernel_size,stride=self.kernel_size).transpose(2,1)
        unfold_input = unfold_input.view(x.shape[0],-1,x.shape[1],self.kernel_size**2).transpose(2,1)
        
        out =torch.max(unfold_input,3)[0].view(x.shape[0],x.shape[1],output_size,output_size)

        return out