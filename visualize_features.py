#!pip install matplotlib

import matplotlib.pyplot as plt
from ResNet18 import ResNet18
import torchvision.datasets as dset
import torchvision.transforms as T
import torch

USE_GPU = True 
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = ResNet18() # replace with the trained model
plt.tight_layout()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

vis_labels = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
for l in vis_labels:

    getattr(model, l).register_forward_hook(get_activation(l))
    
data_dir = './data'
normalize = T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_data = dset.CIFAR10(data_dir, train=False, download=False, transform=normalize)

data, _ = test_data[0]
data = data.unsqueeze_(0).to(device = device, dtype = dtype)
output = model(data)

for idx, l in enumerate(vis_labels):

    act = activation[l].squeeze()
    if idx < 2:
        ncols = 8
    else:
        ncols = 32       
    nrows = act.size(0) // ncols   
    fig, axarr = plt.subplots(nrows, ncols)
    fig.suptitle(l)
    for i in range(nrows):
        for j in range(ncols):
            axarr[i, j].imshow(act[i * nrows + j].cpu())
            axarr[i, j].axis('off')