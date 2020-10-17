#!/usr/bin/env python
# coding: utf-8

# pylint: disable-all
from __future__ import absolute_import
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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


class MultiModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, output_size)
        self.out = nn.Softmax()

    def forward(self, input_):
        out = self.layer1(input_)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.out(out)
        return out


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
    from PyTorch import BinModel, RegModel, MultiModel

    TEST_DATA_ROOT = "~/workspace/datarobot-user-models/tests/testdata"
    BINARY_DATA = os.path.join(TEST_DATA_ROOT, "iris_binary_training.csv")
    REGRESSION_DATA = os.path.join(TEST_DATA_ROOT, "boston_housing.csv")
    MULTICLASS_DATA = os.path.join(TEST_DATA_ROOT, "Skyserver_SQL2_27_2018 6_51_39 PM.csv")

    bin_X = pd.read_csv(BINARY_DATA)
    bin_y = bin_X.pop("Species")

    reg_X = pd.read_csv(REGRESSION_DATA)
    reg_y = reg_X.pop("MEDV")

    multi_X = pd.read_csv(MULTICLASS_DATA)
    multi_y = multi_X.pop("class")

    bin_target_encoder = LabelEncoder()
    bin_target_encoder.fit(bin_y)
    bin_t_X = torch.from_numpy(bin_X.values).type(torch.FloatTensor)
    bin_t_y = torch.from_numpy(bin_target_encoder.transform(bin_y)).type(torch.FloatTensor)
    bin_model = BinModel(bin_X.shape[1])
    bin_opt = optim.Adam(bin_model.parameters(), lr=0.001)
    bin_criterion = nn.BCELoss()

    reg_t_X = torch.from_numpy(reg_X.values).type(torch.FloatTensor)
    reg_t_y = torch.from_numpy(reg_y.values).type(torch.FloatTensor)
    reg_model = RegModel(reg_X.shape[1])
    reg_opt = optim.Adam(reg_model.parameters(), lr=0.001)
    reg_criterion = nn.MSELoss()

    multi_target_encoder = LabelEncoder()
    multi_target_encoder.fit(multi_y)
    multi_t_X = torch.from_numpy(multi_X.values).type(torch.FloatTensor)
    multi_t_y = torch.from_numpy(multi_target_encoder.transform(multi_y)).type(torch.LongTensor)
    multi_model = MultiModel(multi_X.shape[1], len(multi_target_encoder.classes_))
    multi_opt = optim.Adam(multi_model.parameters(), lr=0.001)
    multi_criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    for e in range(num_epochs):
        train_epoch(bin_model, bin_opt, bin_criterion, bin_t_X, bin_t_y)
        train_epoch(reg_model, reg_opt, reg_criterion, reg_t_X, reg_t_y)
        train_epoch(multi_model, multi_opt, multi_criterion, multi_t_X, multi_t_y)

    bin_model.eval()
    reg_model.eval()
    multi_model.eval()

    for model, data in [(bin_model, bin_X), (reg_model, reg_X), (multi_model, multi_X)]:
        data = Variable(
            torch.from_numpy(data.values if type(data) != np.ndarray else data).type(
                torch.FloatTensor
            )
        )
        with torch.no_grad():
            predictions = model(data).cpu().data.numpy()
        print(predictions)

    FIXTURE_ROOT = "~/workspace/datarobot-user-models/tests/fixtures/drop_in_model_artifacts"
    torch.save(bin_model, os.path.expanduser(os.path.join(FIXTURE_ROOT, "torch_bin.pth")))
    torch.save(reg_model, os.path.expanduser(os.path.join(FIXTURE_ROOT, "torch_reg.pth")))
    torch.save(multi_model, os.path.expanduser(os.path.join(FIXTURE_ROOT, "torch_multi.pth")))
