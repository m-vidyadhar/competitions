from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet import gluon, autograd
from mxnet import ndarray as nd

from datetime import datetime
import time
import logging

from util import visualize


__author__ = "Vidyadhar Mudium"


USE_GPU = False
ctx = mx.gpu() if USE_GPU else mx.cpu()


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


class Trainer():
    def __init__(self, train_iter, net, batch_size, latent_z_size) -> None:
        self.train_iter = train_iter
        self.netG = net[0]
        self.netD = net[1]
        self.netFC = net[2]
        self.batch_size = batch_size
        self.latent_z_size = latent_z_size

        # Initilize parameters
        self.netG.initialize(ctx=ctx)
        self.netD.initialize(ctx=ctx)
        self.netFC.initialize(ctx=ctx)

        # Define loss function
        self.loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

        self.real_label = nd.ones((batch_size,), ctx=ctx)
        self.fake_label = nd.zeros((batch_size,), ctx=ctx)
        pass

    def fit(self, epochs, lr, optim="adam", visualize_img=True, verbose=5):
        # Trainer for the generator and the discriminator
        trainerG = gluon.Trainer(self.netG.collect_params(), optim, {"learning_rate": lr})
        trainerD = gluon.Trainer(self.netD.collect_params(), optim, {"learning_rate": lr})
        trainerFC = gluon.Trainer(self.netFC.collect_params(), optim, {"learning_rate": lr})

        # Evaluation Metric
        metric = mx.metric.CustomMetric(facc)

        stamp = datetime.now().strftime("%Y_%m_%d-%H_%M")
        logging.basicConfig(level=logging.INFO)

        for epoch in range(epochs):
            tic = time.time()
            self.train_iter.reset()
            iter = 0
            for batch in self.train_iter:
                data = batch.data[0].as_in_context(ctx)
                label = batch.label[0].as_in_context(ctx)
                latent_z = mx.nd.random_normal(0, 1, shape=(
                    self.batch_size, self.latent_z_size), ctx=ctx)
                G_input = nd.concat(latent_z, label).reshape(-1,
                                                             (self.latent_z_size + 10), 1, 1)

                with autograd.record():
                    # train with real image
                    output = self.netD(data)
                    fc_input = nd.concat(output.reshape((output.shape[0], -1)), label)
                    output = self.netFC(fc_input)
                    errD_real = self.loss(output, self.real_label)
                    metric.update([self.real_label, ], [output, ])

                    # train with fake image
                    fake = self.netG(G_input)
                    output = self.netD(fake.detach())
                    fc_input = nd.concat(output.reshape(
                        (output.shape[0], -1)), label)
                    output = self.netFC(fc_input)
                    errD_fake = self.loss(output, self.fake_label)
                    errD = errD_real + errD_fake
                    errD.backward()
                    metric.update([self.fake_label, ], [output, ])

                trainerD.step(batch.data[0].shape[0])
                trainerFC.step(batch.data[0].shape[0])

                with autograd.record():
                    fake = self.netG(G_input)
                    output = self.netD(fake)
                    fc_input = nd.concat(output.reshape((output.shape[0], -1)), label)
                    output = self.netFC(fc_input)
                    errG = self.loss(output, self.real_label)
                    errG.backward()

                trainerG.step(batch.data[0].shape[0])

                iter = iter + 1
                print("Epoch:", epoch + 1, "; Current batch:",
                      iter, "/ 469", end="\r")
                btic = time.time()

            name, acc = metric.get()
            metric.reset()
            print("\nBinary training accuracy at epoch %d: %s=%f" %
                  (epoch+1, name, acc))
            print("Discriminator Loss: %.3f; Generator Loss: %.3f" % (nd.mean(errD).asscalar(),
                                                                      nd.mean(errG).asscalar()))
            print("time: %s sec" % np.round(time.time() - tic))

            # Visualize one generated image for each epoch
            if visualize_img and (epoch % verbose == 0):
                fake_img = fake[0]
                visualize(fake_img)
                plt.show()
        return

    def generate_samples(self, num_image):
        for digit in range(10):
            for i in range(num_image):
                latent_z = mx.nd.random_normal(
                    0, 1, shape=(1, self.latent_z_size), ctx=ctx
                )
                label = nd.one_hot(nd.array([[digit]]), 10).as_in_context(ctx)
                img = self.netG(nd.concat(latent_z, label.reshape(
                    (1, 10))).reshape(-1, self.latent_z_size+10, 1, 1))
                plt.subplot(2, 5, digit + i + 1)
                visualize(img[0])
        plt.show()
        return