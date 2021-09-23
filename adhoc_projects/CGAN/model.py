from __future__ import print_function
import mxnet as mx
from mxnet.gluon import nn


__author__ = "Vidyadhar Mudium"


def get_networks(model) -> tuple:
    netD, netFC = model.discriminator()
    return (model.generator(), netD, netFC)

class CGAN1():
    """Architecture 1
        1. 4 Layers Convolutions (Transpose) for Generator
        2. 5 Layers Convolutions + 3 Dense Layers for Discriminator
        3. Adam Optimiser
        4. Filter Size for Generator: [4x4]
        5. Filter Size for Discriminator: [3x3]
    """
    def __init__(self) -> None:
        self.w_init = mx.init.Xavier()
        pass


    def generator(self):
        # Build the generator
        netG = nn.HybridSequential()
        with netG.name_scope():
            netG.add(nn.Conv2DTranspose(64*2, 4, 1, 0, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(64, 4, 1, 0, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(32, 4, 2, 1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(1, 4, 2, 1, use_bias=False))
            netG.add(nn.Activation("sigmoid"))
        return netG
    

    def discriminator(self) -> tuple:
        # Build the discriminator
        netD = nn.HybridSequential()
        with netD.name_scope():
            netD.add(nn.Conv2D(16, 3, 1, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(32, 3, 2, 1, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(32, 3, 2, 1, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(32, 3, 2, 1, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(64, 4, 1, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.Flatten())
        
        netFC = nn.HybridSequential()
        with netFC.name_scope():
            netFC.add(nn.Dense(units=64, activation="relu", weight_initializer=self.w_init))
            netFC.add(nn.Dense(units=64, activation="relu", weight_initializer=self.w_init))
            netFC.add(nn.Dense(units=1, weight_initializer=self.w_init))
        return (netD, netFC)


class CGAN2():
    """Architecture 2 - Filters & Convolutions
        1. **6 Layers Convolutions (Transpose) for Generator**
        2. **6 Layers Convolutions + 3 Dense Layers for Discriminator**
        3. Adam Optimiser
        4. **Filter Size for Generator: [3x3]**
        5. **Filter Size for Discriminator: [4x4]**
    """
    def __init__(self) -> None:
        self.w_init = mx.init.Xavier()
        pass


    def generator(self):
        # Build the generator
        netG = nn.HybridSequential()
        with netG.name_scope():
            netG.add(nn.Conv2DTranspose(64*2, 3, 1, 0, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(64, 3, 1, 1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(64, 3, 2, 1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(32, 3, 2, 1, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(32, 3, 2, 2, use_bias=False))
            netG.add(nn.BatchNorm())
            netG.add(nn.Activation("relu"))
            netG.add(nn.Conv2DTranspose(1, 4, 2, 2, use_bias=False))
            netG.add(nn.Activation("sigmoid"))
        return netG
    

    def discriminator(self):
        # Build the discriminator
        netD = nn.HybridSequential()
        with netD.name_scope():
            netD.add(nn.Conv2D(16, 4, 1, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(32, 3, 1, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(32, 3, 1, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(64, 4, 2, 1, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(64, 4, 2, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.BatchNorm())
            netD.add(nn.LeakyReLU(0.2))
            netD.add(nn.Conv2D(1, 4, 2, 0, use_bias=False, weight_initializer=self.w_init))
            netD.add(nn.Flatten())

        netFC = nn.HybridSequential()
        with netFC.name_scope():
            netFC.add(nn.Dense(units = 64, activation = "relu", weight_initializer=self.w_init))
            netFC.add(nn.Dense(units = 64, activation = "relu", weight_initializer=self.w_init))
            netFC.add(nn.Dense(units=1, weight_initializer=self.w_init))
        return (netD, netFC)