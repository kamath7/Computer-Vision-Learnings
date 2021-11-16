from __future__ import print_function
import torch
import torch.nn as nn 
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset 
import torchvision.transforms as transforms 
import torchvision.utils as vutils 
from torch.autograd import Variable


#parameters 
batchSize = 69 
imageSize = 69

transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),]) #transforming ip images

#reading dataset . we use CIFAR for training

dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.dataloader(dataset, batch_size = batchSize, shuffle=True, num_workers=2)