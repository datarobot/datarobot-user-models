from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


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


class PytorchRegressor(BaseEstimator, RegressorMixin):
    """A pytorch regressor"""

    def __init__(self, n_epochs):
        """
        Called when initializing the regressor
        """
        self.n_epochs = n_epochs
        self.reg_model = None
        self.reg_opt = None
        self.reg_criterion = None

    def _build_model(self, X):
        reg_model = RegModel(X.shape[1])
        reg_opt = optim.Adam(reg_model.parameters(), lr=0.001)
        reg_criterion = nn.MSELoss()

        return reg_model, reg_opt, reg_criterion

    def _train_model(self, X, y):
        reg_t_X = torch.from_numpy(X).type(torch.FloatTensor)
        reg_t_y = torch.from_numpy(y.values).type(torch.FloatTensor).reshape(-1, 1)

        for e in range(self.n_epochs):
            train_epoch(self.reg_model, self.reg_opt, self.reg_criterion, reg_t_X, reg_t_y)

    def fit(self, X, y):
        """
        Trains the pytorch regressor.
        """

        self.reg_model, self.reg_opt, self.reg_criterion = self._build_model(X)
        self._train_model(X, y)

    def predict(self, X):
        """
        Makes a prediction using the trained pytorch model
        """
        reg_t_X = torch.from_numpy(X).type(torch.FloatTensor)
        return self.reg_model(reg_t_X).data.numpy()


class PytorchClassifier(BaseEstimator, ClassifierMixin):
    """A pytorch regressor"""

    def __init__(self, n_epochs):
        """
        Called when initializing the regressor
        """
        self.n_epochs = n_epochs
        self.bin_model = None
        self.bin_opt = None
        self.bin_criterion = None

    def _build_model(self, X):
        bin_model = BinModel(X.shape[1])
        bin_opt = optim.Adam(bin_model.parameters(), lr=0.001)
        bin_criterion = nn.BCELoss()

        return bin_model, bin_opt, bin_criterion

    def _train_model(self, X, y):
        target_encoder = LabelEncoder()
        target_encoder.fit(y)
        bin_t_X = torch.from_numpy(X).type(torch.FloatTensor)
        bin_t_y = torch.from_numpy(target_encoder.transform(y)).type(torch.FloatTensor)

        for e in range(self.n_epochs):
            train_epoch(self.bin_model, self.bin_opt, self.bin_criterion, bin_t_X, bin_t_y)

    def fit(self, X, y):
        """
        Trains the pytorch classifier.
        """

        self.bin_model, self.bin_opt, self.bin_criterion = self._build_model(X)
        self._train_model(X, y)

    def predict(self, X):
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=37825, stdoutToServer=True, stderrToServer=True)
        bin_t_X = torch.from_numpy(X).type(torch.FloatTensor)
        return self.bin_model(bin_t_X).data.numpy()

