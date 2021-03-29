# Convolutional Neural Networks from scratch
Custom implementation of CNN layers without using torch.nn modules and ResNet-18 with ```Pytorch```on the CIFAR-10 dataset.

## Project Structure 
### Custom CNN Layers

| Custom Layers          | Function          | 
| ------------- |:-------------:| 
| [BatchNorm2D.py](https://github.com/Nasmasim/modular-CNNs/blob/main/custom_cnn_layers/BatchNorm2D.py)| Batch Normalization over a mini-batch of 2D inputs |
| [MaxPool2D.py](https://github.com/Nasmasim/modular-CNNs/blob/main/custom_cnn_layers/MaxPool2D.py)      | Max-pooling layer      |
| [conv2D.py](https://github.com/Nasmasim/modular-CNNs/blob/main/custom_cnn_layers/conv2D.py) | Convolutional layer      |
| [linear.py](https://github.com/Nasmasim/modular-CNNs/blob/main/custom_cnn_layers/linear.py) | Linear layer |

### ResNet 18 
Implementation of the Deep Residual Learning architecture by He, Kaiming, et al. in the paper ["Deep Residual Learning for Image Recognition"](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). In a residual network, each block contains some convolutional layers, plus "skip" connections", which allow the activation to by pass a layer and then be summed up with the activations of the skipped layer. The ResNet-18 in [ResNet18.py](https://github.com/Nasmasim/modular-CNNs/blob/main/ResNet18.py) is trained on the CIFAR-10 dataset. 

### Visualising Feature maps 
[visualize_features.py](https://github.com/Nasmasim/modular-CNNs/blob/main/visualize_features.py) allows to visualise feature maps computed by different layers of the network. 
<table>
  <tr>
    <td>conv1</td>
     <td>layer1</td>
     <td>layer2</td>
  </tr>
  <tr>
    <td><img src="https://github.com/Nasmasim/modular-CNNs/blob/main/figures/feature1.png" width="90%"></td>
    <td><img src="https://github.com/Nasmasim/modular-CNNs/blob/main/figures/feature2.png" width="90%"></td>
    <td><img src="https://github.com/Nasmasim/modular-CNNs/blob/main/figures/feature3.png" width="90%"></td>
  </tr>
  <tr>
    <td>layer3</td>
     <td>layer4</td>
  </tr>
  <tr>
    <td><img src="https://github.com/Nasmasim/modular-CNNs/blob/main/figures/feature4.png" width="90%"></td>
    <td><img src="https://github.com/Nasmasim/modular-CNNs/blob/main/figures/feature5.png" width="90%"></td>
  </tr>
 </table>
