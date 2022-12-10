"""
File: module.py
Desc: Basic modules for developing statistic or gradient-based
      classifiers by using numpy or pytorch.
"""
import numpy as np
import torch as tc


class StatisticClassifier:
    def __init__(self, name):
        self.name = name
        self._num_feat = None
        self._num_cls = None

    def fit(self, X, Y):
        assert len(X.shape) == 2 and len(Y.shape) == 1 and X.shape[0] == Y.shape[0]
        self._num_feat = X.shape[1]
        self._num_cls = len(np.unique(Y))
        assert max(Y) == self._num_cls - 1
        self._fit(X, Y)

    def predict(self, X):
        assert self._num_feat and self._num_cls, "call model.fit() before prediction"
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        assert len(X.shape) == 2 and X.shape[1] == self._num_feat
        return np.apply_along_axis(self.forward, 1, X)


class GradientClassifier(tc.nn.Module):
    def __init__(self, name, device="cpu"):
        tc.nn.Module.__init__(self)
        self.name = name
        self._device = device

    def _train_epoch(self, dataset, opt, lossfn):
        self.train()
        self.to(self._device)
        for batch in dataset:
            if len(batch.shape) == 1:
                batch = batch.unsqueeze(0)
            x, y = batch[:, :-1].to(self._device).float(), batch[:, -1].to(self._device)
            opt.zero_grad()
            lossfn(self(x).squeeze(), y.squeeze()).backward()
            opt.step()

    def _test_epoch(self, dataset):
        model.eval()
        model.to(self._device)
        Y, P = [], []
        for batch in dataset:
            if len(batch.shape) == 1:
                batch = batch.unsqueeze(0)
            Y.append(batch[:, -1].squeeze().numpy())
            P.append(self(batch[:, :-1].float().to(self._device)).cpu().detach().squeeze().numpy())
        return scoring(np.hstack(Y), np.hstack(P))

    def fit(self, X, Y, epochs=1000, learn_rate=1e-3, batch_size=16):
        assert len(X.shape) == 2 and len(Y.shape) == 1 and X.shape[0] == Y.shape[0]
        dataset = tc.utils.data.DataLoader(np.hstack([X, Y.reshape((-1, 1))]),
                                           batch_size=batch_size, shuffle=True)
        optimizer = tc.optim.SGD(self.parameters(), lr=learn_rate)
        lossfunc = tc.nn.BCELoss()
        for epoch in range(1, epochs + 1):
            self._train_epoch(dataset, optimizer, lossfunc)
        
    def predict(self, X, batch_size=16):
        self.eval()
        y_hat = []
        addone = 1 if len(X) % batch_size != 0 else 0
        with tc.no_grad():
            for idx in range(len(X) // batch_size + addone):
                batch = tc.tensor(X[idx * batch_size: (idx + 1) * batch_size])
                y_hat.append(self(batch.to(self._device)).cpu().squeeze().numpy())
        return np.hstack(y_hat)
