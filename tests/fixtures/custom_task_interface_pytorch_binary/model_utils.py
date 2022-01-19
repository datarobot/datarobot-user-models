"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
#!/usr/bin/env python
# coding: utf-8

# pylint: disable-all
from __future__ import absolute_import
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


class BinModel(nn.Module):
    expected_target_type = torch.FloatTensor

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


def train_epoch(model, opt, criterion, X, y, batch_size=50):
    batch_size = int(sqrt(X.shape[0]))
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i : beg_i + batch_size, :]
        # y_hat will be (batch_size, 1) dim, so coerce target to look the same
        y_batch = y[beg_i : beg_i + batch_size].reshape(-1, 1)
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


def build_classifier(X, num_labels):
    class_model = BinModel(X.shape[1])
    class_opt = optim.Adam(class_model.parameters(), lr=0.001)
    class_criterion = nn.BCELoss()

    return class_model, class_opt, class_criterion


def train_classifier(X, y, class_model, class_opt, class_criterion, n_epochs=10):
    target_encoder = LabelEncoder()
    target_encoder.fit(y)
    transformed_y = target_encoder.transform(y)
    bin_t_X = torch.from_numpy(X.values).type(torch.FloatTensor)
    bin_t_y = torch.from_numpy(transformed_y).type(class_model.expected_target_type)

    for e in range(n_epochs):
        train_epoch(class_model, class_opt, class_criterion, bin_t_X, bin_t_y)
