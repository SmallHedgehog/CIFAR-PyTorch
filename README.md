## CIFAR-PyTorch
train CIFAR10 with PyTorch(0.4.0)

## Implemented CNNs
* LeNet
* VGG
* GoogLeNet
* ResNet
* PreActResNet
* ResNeXt
* DenseNet
* MobileNet
* ShuffleNet
* DPN
* SeNet

## Usage
```
usage: main.py [-h] [--net NET] [--lr LR] [--epoch EPOCH] [--device DEVICE]

train CIFAR10　with PyTorch

optional arguments:
  -h, --help       show this help message and exit
  --net NET        choose a network
  --lr LR          learning rate
  --epoch EPOCH    epochs
  --device DEVICE  choose device(gpu or cpu)
```
#### example
```
python main.py --net LeNet --lr 0.01 --epoch 100 --device gpu
```
