#!/usr/bin/env python
# coding: utf-8

# pylint: disable-all
from __future__ import absolute_import
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


class BinModel(nn.Module):
    def __init__(self, input_size):
        super(BinModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


class RegModel(nn.Module):
    def __init__(self, input_size):
        super(RegModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        y = self.out(h2)
        return y


def train_epoch(model, opt, criterion, X, y, batch_size=50):
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i : beg_i + batch_size, :]
        y_batch = y[beg_i : beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = model(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses


if __name__ == "__main__":
    from PyTorch import BinModel, RegModel

    BINARY_DATA = "iris_binary_training.csv"
    REGRESSION_DATA = "boston_housing.csv"

    bin_X = pd.read_csv(BINARY_DATA)
    bin_y = bin_X.pop("Species")

    reg_X = pd.read_csv(REGRESSION_DATA)
    reg_y = reg_X.pop("MEDV")

    target_encoder = LabelEncoder()
    target_encoder.fit(bin_y)
    bin_t_X = torch.from_numpy(bin_X.values).type(torch.FloatTensor)
    bin_t_y = torch.from_numpy(target_encoder.transform(bin_y)).type(torch.FloatTensor)
    bin_model = BinModel(bin_X.shape[1])
    bin_opt = optim.Adam(bin_model.parameters(), lr=0.001)
    bin_criterion = nn.BCELoss()

    reg_t_X = torch.from_numpy(reg_X.values).type(torch.FloatTensor)
    reg_t_y = torch.from_numpy(reg_y.values).type(torch.FloatTensor)
    reg_model = RegModel(reg_X.shape[1])
    reg_opt = optim.Adam(reg_model.parameters(), lr=0.001)
    reg_criterion = nn.MSELoss()

    num_epochs = 20
    for e in range(num_epochs):
        train_epoch(bin_model, bin_opt, bin_criterion, bin_t_X, bin_t_y)
        train_epoch(reg_model, reg_opt, reg_criterion, reg_t_X, reg_t_y)

    bin_model.eval()
    reg_model.eval()

    torch.save(bin_model, "torch_bin.pth")
    torch.save(reg_model, "torch_reg.pth")
