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
from math import sqrt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


class MultiModel(nn.Module):
    expected_target_type = torch.LongTensor

    def __init__(self, input_size, output_size):
        super(MultiModel, self).__init__()
        hidden_size = max(input_size, output_size) * 2
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.out = nn.Softmax()

    def forward(self, input_):
        out = self.layer1(input_)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.out(out)
        return out


def train_epoch(model, opt, criterion, X, y):
    batch_size = int(sqrt(X.shape[0]))
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i : beg_i + batch_size, :]
        # y_hat will be (batch_size, 1) dim, so coerce target to look the same
        y_batch = y[beg_i : beg_i + batch_size].reshape(-1, 1).flatten()
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
    class_model = MultiModel(X.shape[1], num_labels)
    class_opt = optim.Adam(class_model.parameters(), lr=0.001)
    class_criterion = nn.CrossEntropyLoss()

    return class_model, class_opt, class_criterion


def train_classifier(X, y, class_model, class_opt, class_criterion, n_epochs=10):
    bin_t_X = torch.from_numpy(X.values).type(torch.FloatTensor)
    bin_t_y = torch.from_numpy(y).type(class_model.expected_target_type)

    for e in range(n_epochs):
        train_epoch(class_model, class_opt, class_criterion, bin_t_X, bin_t_y)
