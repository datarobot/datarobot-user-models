#!/usr/bin/env python
# coding: utf-8

# pylint: disable-all
from __future__ import absolute_import
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

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
        y_hat = model(x_batch).squeeze(1)  # ensure y_hat dimension is the same as y_batch dim
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses


def build_classifier(X):
    bin_model = BinModel(X.shape[1])
    bin_opt = optim.Adam(bin_model.parameters(), lr=0.001)
    bin_criterion = nn.BCELoss()

    return bin_model, bin_opt, bin_criterion


def build_regressor(X):
    reg_model = RegModel(X.shape[1])
    reg_opt = optim.Adam(reg_model.parameters(), lr=0.001)
    reg_criterion = nn.MSELoss()

    return reg_model, reg_opt, reg_criterion


def train_classifier(X, y, bin_model, bin_opt, bin_criterion, n_epochs=5):
    target_encoder = LabelEncoder()
    target_encoder.fit(y)
    bin_t_X = torch.from_numpy(X.values).type(torch.FloatTensor)
    bin_t_y = torch.from_numpy(target_encoder.transform(y)).type(torch.FloatTensor)

    for e in range(n_epochs):
        train_epoch(bin_model, bin_opt, bin_criterion, bin_t_X, bin_t_y)


def train_regressor(X, y, reg_model, reg_opt, reg_criterion, n_epochs=5):
    reg_t_X = torch.from_numpy(X.values).type(torch.FloatTensor)
    reg_t_y = torch.from_numpy(y.values).type(torch.FloatTensor)

    for e in range(n_epochs):
        train_epoch(reg_model, reg_opt, reg_criterion, reg_t_X, reg_t_y)


def save_torch_model(model, output_dir_path, filename="torch_bin.pth"):
    output_file_path = Path(output_dir_path) / filename
    torch.save(model, output_file_path)
