# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # self.model = nn.Sequential(
        #     nn.Linear(in_size, 20),
        #     nn.LeakyReLU(),
        #     nn.Linear(20, 50),
        #     nn.LeakyReLU(),
        #     nn.Linear(50, 50),
        #     nn.LeakyReLU(),
        #     nn.Linear(50, out_size)
        # )
        self.model = nn.Sequential(
            nn.Unflatten(1, (3, 32, 32)),
            nn.Conv2d(3, 6, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 3, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, (3, 3)),
            nn.Flatten(),
            nn.BatchNorm1d(1728),
            nn.Linear(1728, out_size)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=lrate)


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        norm_x = (x - mean) / std
        y = self.model(norm_x)
        return y

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        y_hat = self.model(x)
        # print(y_hat.shape)
        loss = self.loss_fn(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    net = NeuralNet(0.02, nn.CrossEntropyLoss(), train_set.shape[1], 2)
    # mean = train_set.mean(dim=1, keepdim=True)
    # std = train_set.std(dim=1, keepdim=True)
    # normalized_dev = (train_set - mean) / std
    batch_list = torch.split(train_set, batch_size)
    batch_labels = torch.split(train_labels, batch_size)
    losses = []
    for i in range(n_iter):
        batch_in = batch_list[i % len(batch_list)]
        batch_label_in = batch_labels[i % len(batch_list)]
        mean = batch_in.mean(dim=1, keepdim=True)
        std = batch_in.std(dim=1, keepdim=True)
        normalized_set = (batch_in - mean) / std
        loss = net.step(normalized_set, batch_label_in)
        losses.append(loss)

    outputs = net.forward(dev_set)
    _, pred = torch.max(outputs, 1)
    loss_ret = np.array(losses)

    return loss_ret, np.array(pred), net
