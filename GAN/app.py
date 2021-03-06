from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
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

# creating a discriminator class


class D (nn.Module):

    def __init__(self):
        super(D, self).__init__()
        # creating a nn for Discriminator too
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )  # Leaky relu has negative slope compared to Relu

    # input is an image. op will be a val bw 0 and 1. 0 reject image, 1 accept iamge
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)  # flattening for the op to be 1D


neural_D = D()
neural_D.apply(weights_init)

# training our d and G

criterion = nn.BCELoss()  # defining loss. BCE -> Binary CrossEntropy
optimiserD = optim.Adam(neural_D.parameters(), lr=0.0002,
                        betas=(0.5, 0.999))  # optimiser for D
optimiserG = optim.Adam(neuralG.parameters(), lr=0.0002,
                        betas=(0.5, 0.999))  # optimiser for G

if __name__ == '__main__':
    for epoch in range(25):
        for i, data in enumerate(dataloader, 0):
            # Discriminator portion
            # updating weights of discriminator
            neural_D.zero_grad()
            # training discriminator to discriminate by giving it a real image of dataset
            real, _ = data
            input = Variable(real)
            # size of the mini batch. 1s why? because our targets are real images
            target = Variable(torch.ones(input.size()[0]))
            output = neural_D(input)
            # calculating loss error for training discriminator to understand what is real
            err_D_real = criterion(output, target)
            # training discriminator to discriminate by giving it a fake image of dataset
            # 100 because in the NN we have mentioned 100 feature maps
            noise = torch.randn(input.size()[0], 100, 1, 1)
            fake = neuralG(noise)
            # size of the mini batch. 0s why? because our targets are fake images
            target = Variable(torch.zeros(input.size()[0]))
            output = neural_D(fake.detach())
            # calculating loss error for training discriminator to understand what is fake
            err_D_fake = criterion(output, target)

            # backpropagation of the error for the discriminator to manage weights again
            errD = err_D_real + err_D_fake
            errD.backward()
            optimiserD.step()

            # Generator portion
            # update weights of nn of generator
            neuralG.zero_grad()
            target = Variable(torch.ones(input.size()[0]))
            output = neural_D(fake)
            errG = criterion(output, target)
            errG.backward()
            optimiserG.step()

            # showing the losses and saving real imgs along with generated
            print('[%d/%d][%d/%d] Loss of Discriminator: %.4f Loss of Generator: %.4f' %
                  (epoch, 25, i, len(dataloader), errD.data, errG.data))
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' %
                                  "./results", normalize=True)
                fake = neuralG(noise)
                vutils.save_image(fake.data, '%s/fake_sample_epoch_%03d.png' %
                                  ("./results", epoch), normalize=True)
