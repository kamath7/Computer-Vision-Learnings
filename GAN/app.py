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


# parameters
batchSize = 69
imageSize = 69

transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(
), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])  # transforming ip images

# reading dataset . we use CIFAR for training

dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batchSize, shuffle=True, num_workers=2)


def weights_init(m):  # here input is a neural net and weights will be set here. looks at layers and does as mentioned
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)

# defining the generator class. Inherited nn.module to create neural net


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        # meta module for all different layers/methods. Inverse CNN is created here
        self.main = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(
            512), nn.ReLU(True), nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(
            256), nn.ReLU
            (True), nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(
            128), nn.ReLU(True),  nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  nn.BatchNorm2d(
            64), nn.ReLU(True), nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh()
        )  # defining layers of the nn. 3 channels

    def forward(self, input):  # forward propagation to Discriminatory
        output = self.main(input)
        return output  # output of the generator


neuralG = G()  # creating an instance
neuralG.apply(weights_init)
