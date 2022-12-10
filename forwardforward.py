import random
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FullyConnected:
    
    def __init__(self, in_dim, out_dim, do_norm=True, do_active=True, threshold=1.0, bias=True):
        k = np.sqrt(1.0 / in_dim)
        self._W = np.random.uniform(-k, k, (in_dim, out_dim))
        self._b = np.random.uniform(-k, k, out_dim) if bias else np.zeros(out_dim)
        self._gradW, self._gradb = 0.0, 0.0
        self._theta = threshold
        self._do_norm = do_norm
        self._do_act = do_active
        self._samples = 0.0
        self.training = True

    @property
    def training(self):
        return self._is_training

    @training.setter
    def training(self, mode):
        assert mode in (True, False)
        self._is_training = mode
        self._cache = None

    def __call__(self, X):
        return self.forward(X)

    def _normalize(self, X):
        """normalize the inputs from previous layers
        Math: X / ||X||_2
        Ref: ``To prevent this, FF normalizes the length of the
               hidden vector before using it as input to the
               next layer.``
        """
        if self._do_norm:
            return X / (1e-9 + (X ** 2).sum(axis=-1, keepdims=True))
        return X

    def _goodness(self, H):
        """compute the goodness score.
        Math: \sum_{d=1}^D H_d
        Ref: ``Let us suppose that the goodness function for a layer
               is simply the sum of the squares of the activities of
               the rectified linear neurons in that layer.``
        """
        return (H ** 2).sum(axis=-1)

    def _proba(self, H):
        """compute the probability of sample labels.
        Math: \sigmoid(goodness - \theta)
        Ref: ``the probability that an input vector is positive is
               given by applying the logistic function \sigmoid to
               the goodness, minus some threshold \theta.``
        """
        if self._do_act:
            return self._goodness(H)
        return H

    def forward(self, X):
        assert X.shape[-1] == self._W.shape[0]
        X = self._normalize(X)
        h = X @ self._W + self._b
        if self.training:
            self._cache = (X, h,)
        return self._proba(h)

    def backward(self):
        x, h = self._cache
        self._samples += x.shape[0]
        y = sigmoid(self._goodness(h) - self._theta)
        
        grady = y * (1 - y)
        gradh = 2 * grady.reshape(-1, 1) * h
        self._gradW += x.T @ gradh       # (indim, outdim)
        self._gradb += gradh.sum(axis=0) # (outdim,)

    def update(self, positive, learn_rate):
        assert positive in (True, False)
        sign = 1.0 if positive else -1.0
        self._W += sign * learn_rate * self._gradW #/ self._samples
        self._b += sign * learn_rate * self._gradb #/ self._samples
        self._gradW, self._gradb, self._samples = 0.0, 0.0, 0.0
        

class ForwardForwardClassifier:
    name = "ForwardForwardNetwork"
    def __init__(self, in_dim=4, hide_dim=200, out_dim=2, n_layers=2):
        self._dims = (in_dim, hide_dim, out_dim)
        self.layers = []
        for layer in range(n_layers):
            do_norm = layer > 0               # only the first layer does not require normalization
            do_active = layer == n_layers - 1 # only the last layer should do activation for outputs
            input_dim = in_dim + out_dim if layer == 0 else hide_dim
            self.layers.append(FullyConnected(input_dim, hide_dim, do_norm, do_active))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self):
        for layer in self.layers:
            layer.backward()

    def update(self, positive, learn_rate):
        for layer in self.layers:
            layer.update(positive, learn_rate)

    def fit(self, X, Y, learn_rate=0.05, batch_size=64, epochs=100):
        import time
        from sklearn.metrics import accuracy_score
        Y = Y.astype(np.int32)
        self._labels = ylist = set(Y.tolist())
        assert len(ylist) <= self._dims[-1]
        assert all((isinstance(y, int) for y in ylist)), "labels should be integers"
        assert all((0 <= y < self._dims[-1] for y in ylist)), "labels should be indicates between 0 and %d" % (self._dims[-1],)

        for epoch in range(epochs):
            begin = time.time()
            for y in ylist:
                subY = list(ylist - {y})
                subX = X[Y == y]
                for batch in self._generate_batches(subX, batch_size):
                    real_y = np.zeros((len(batch), len(ylist)))
                    real_y[:, y] = 1
                    self.forward(np.hstack([batch, real_y]))
                    self.backward()
                self.update(positive=True, learn_rate=learn_rate)

                for batch in self._generate_batches(subX, batch_size):
                    fake_y = np.zeros((len(batch), len(ylist)))
                    fake_y[range(len(batch)), random.choices(subY, k=len(batch))] = 1
                    self.forward(np.hstack([batch, fake_y]))
                    self.backward()
                self.update(positive=False, learn_rate=learn_rate)
            Yhat = self.predict(X, batch_size)
            if epoch % 10 == 0:
                print("Epoch-%d | Spent=%.4f | Accuracy=%.4f" % (epoch, time.time() - begin, accuracy_score(Y, Yhat)))

    def predict(self, X, batch_size=32):
        Yhat = []
        for batch in self._generate_batches(X, batch_size):
            batch_yhat = []
            for y in self._labels:
                temp_y = np.zeros((len(batch), len(self._labels)))
                temp_y[:, y] = 1
                temp_x = np.hstack([batch, temp_y])
                pred_y = self.forward(temp_x)
                batch_yhat.append(pred_y.reshape(-1, 1))
            Yhat.extend(np.argmax(np.hstack(batch_yhat), -1))
        return np.array(Yhat)

    def _generate_batches(self, X, batch_size=32):
        batch = []
        for sample in X:
            batch.append(sample)
            if len(batch) == batch_size:
                yield np.vstack(batch)
                batch.clear()
        if len(batch) > 0:
            yield np.vstack(batch)
        

if __name__ == "__main__":
    X = np.random.normal(size=(100, 5))
    Y = X[:, 0] * 5 + X[:, 1] * 4 + X[:, 2] * -8 + X[:, 3] * -0.5 + X[:,4] * 0.0
    Y = np.where(Y >= 0, 1, 0)
    net = ForwardForwardClassifier(5, 100, 2, n_layers=2)
    net.fit(X, Y)

