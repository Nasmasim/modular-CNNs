import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):

        super(Conv2d, self).__init__()
        """
        An implementation of a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Parameters:
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - kernel_size: Size of the convolving kernel
        - stride: The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - padding: The number of pixels that will be used to zero-pad the input.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding
        
        # weight and bias initialisation
        self.w = nn.Parameter(torch.Tensor(out_channels, in_channels, 
                                                 kernel_size, kernel_size))
        self.w.data.normal_(-0.1, 0.1)
        self.w = nn.init.kaiming_normal(self.w)
        
        #check if bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, ))
            self.bias.data.normal_(-0.1, 0.1)
            self.bias = nn.init.kaiming_normal(self.bias)
          
        else:
            self.bias = None
            
    def forward(self, x):
        """
        Input:
        - x: Input data of shape (N, C, H, W)
        Output:
        - out: Output data, of shape (N, F, H', W').
        """

        self.N = x.shape[0]
         
        H = x.shape[2]  
        output_size = int((H - self.kernel_size + 2 * self.padding) / self.stride) + 1
        unfold_input = F.unfold(x,self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)       
        unfold_output = unfold_input.transpose(1, 2) @ self.w.view(self.out_channels, -1).transpose(1, 0)
        
        if self.bias != None:
            unfold_output += self.bias
        out = unfold_output.transpose(1,2).view(self.N, self.out_channels, output_size, output_size)
        
        return out
