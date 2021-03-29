import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

from ResNet18 import ResNet18

# preprocessing functions
augment = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(32, padding=4)])
normalize = T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_size = 49000

# split the data set into train validation and test sets


data_dir = './data'
train_data = dset.CIFAR10(data_dir, train=True, download=True, transform=T.Compose([augment,normalize]))
loader_train = DataLoader(train_data, batch_size=64, sampler=sampler.SubsetRandomSampler(range(train_size)))

val_data = dset.CIFAR10(data_dir, train=True, download=True, transform=normalize)
loader_val = DataLoader(val_data, batch_size=64, sampler=sampler.SubsetRandomSampler(range(train_size, 50000)))

test_data = dset.CIFAR10(data_dir, train=False, download=True, transform=normalize)
loader_test = DataLoader(test_data, batch_size=64)

USE_GPU = True 
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
print_every = 100
def check_accuracy(loader, model):
    # function for test accuracy on validation and test set
    
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        
# added loader as parameter
def train_part(loader, model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(len(loader_train))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epochs: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                #check_accuracy(loader_val, model)
                #print()
                # define and train the network

model = ResNet18()
optimizer = optim.Adam(model.parameters(), lr=0.000182, weight_decay=0.001881)

train_part(loader_train, model, optimizer, epochs = 10)

# report test set accuracy
check_accuracy(loader_test, model)

# save the model
torch.save(model.state_dict(), 'model.pt')

# load the model
model = ResNet18()
model.load_state_dict(torch.load('model.pt'))
model = model.to(device=device)
model.eval()

# report test set accuracy
check_accuracy(loader_test, model)



