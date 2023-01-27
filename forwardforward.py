import random
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def norm(x, p=2, dim=-1, keepdims=True):
    return (x ** p).sum(axis=dim, keepdims=keepdims)


class FullyConnected:
    
    def __init__(self, in_dim, out_dim, do_norm=True, constant=1.0):
        k = np.sqrt(1.0 / in_dim)
        self._W = np.random.uniform(-k, k, (in_dim, out_dim))
        self._b = np.random.uniform(-k, k, out_dim)
        self._c = constant
        self._gradW, self._gradb = 0.0, 0.0
        self._do_normalize = do_norm
        self._samples = 0.0
        self.training = True

    @property
    def training(self):
        return self._is_training

    @training.setter
    def training(self, mode):
        assert isinstance(mode, bool)
        self._is_training = mode

    def __call__(self, X):
        return self.forward(X)

    def _normalize(self, X, epsilon=1e-9):
        """
        normalize the inputs from previous layers
        Math: X / ||X||_2
        Ref: ``To prevent this, FF normalizes the length of the
               hidden vector before using it as input to the
               next layer.``
        """
        return X / (epsilon + norm(X, keepdims=True) ** 0.5)

    def goodness(self, H):
        """
        compute the goodness score.
        Math: \sum_{d=1}^D H_d
        Ref: ``Let us suppose that the goodness function for a layer
               is simply the sum of the squares of the activities of
               the rectified linear neurons in that layer.``
        """
        return norm(H)

    def _backward(self, x, h):
        if self.training:
            self._samples += x.shape[0]
            y = sigmoid(self.goodness(h) - self._c)
            grad_y = y * (1 - y)
            grad_h = 2 * grad_y.reshape(-1, 1) * h
            self._gradW += x.T @ grad_h        # (indim, outdim)
            self._gradb += grad_h.sum(axis=0)  # (outdim,)

    def _transform(self, X):
        assert X.shape[-1] == self._W.shape[0]
        return np.maximum(X @ self._W + self._b, 0)

    def forward(self, X):
        if self._do_normalize:
            X = self._normalize(X)
        h = self._transform(X)
        self._backward(X, h)
        return h

    def update(self, is_positive, learning_rate):
        assert isinstance(is_positive, bool)
        sign = 1.0 if is_positive else -1.0
        self._W += sign * learning_rate * self._gradW / self._samples
        self._b += sign * learning_rate * self._gradb / self._samples
        self._gradW, self._gradb, self._samples = 0.0, 0.0, 0.0
        

class ForwardForwardClassifier:
    
    name = "ForwardForwardNetwork"
    
    def __init__(self, in_dim, hide_dim, out_dim, n_layers=2):
        assert n_layers >= 1
        self._dims = (in_dim, hide_dim, out_dim)
        self.layers = [FullyConnected(in_dim + out_dim, hide_dim, False)]
        for layer in range(1, n_layers):
            self.layers.append(FullyConnected(hide_dim, hide_dim, True))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return layer.goodness(X)

    def train_step(self, positive, negative, learning_rate, n_iters=5):
        for layer in self.layers:
            for niter in range(n_iters):
                layer(positive)
                layer.update(is_positive=True, learning_rate=learning_rate)
                layer(negative)
                layer.update(is_positive=False, learning_rate=learning_rate)
            positive = layer(positive)
            negative = layer(negative)

    def fit(self, X, Y, learn_rate=5e-4, batch_size=32, epochs=10000, log_freq=1000):
        from time import time
        from sklearn.metrics import accuracy_score
        Y = Y.astype(np.int32)
        self._labels = ylist = set(Y.tolist())
        assert len(ylist) <= self._dims[-1]
        assert all((isinstance(y, int) for y in ylist)), "labels should be integers"
        assert all((0 <= y < self._dims[-1] for y in ylist)), "labels should fall between 0 and %d" % (self._dims[-1],)

        begin = time()
        for epoch in range(epochs):
            self.training = True
            for y in ylist:
                subY = list(ylist - {y})
                subX = X[Y == y]
                for batch in self._generate_batches(subX, batch_size):
                    real_y = np.zeros((len(batch), len(ylist)))
                    real_y[:, y] = 1
                    fake_y = np.zeros((len(batch), len(ylist)))
                    fake_y[range(len(batch)), random.choices(subY, k=len(batch))] = 1
                    self.train_step(positive=np.hstack([batch, real_y]),
                                    negative=np.hstack([batch, fake_y]),
                                    learn_rate=learn_rate)

            self.training = False
            Yhat = self.predict(X, batch_size)
            if epoch % log_freq == 0:
                acc = accuracy_score(Y, Yhat)
                print("Epoch-%d | Spent=%.4f | Train Accuracy=%.4f" % (epoch, time() - begin, acc))
                begin = time()

    def predict_proba(self, X, batch_size=32):
        Yhat = []
        for batch in self._generate_batches(X, batch_size):
            batch_yhat = []
            for y in self._labels:
                temp_y = np.zeros((len(batch), len(self._labels)))
                temp_y[:, y] = 1
                pred_y = self.forward(np.hstack([batch, temp_y]))
                batch_yhat.append(pred_y.reshape(-1, 1))
            Yhat.append(np.hstack(batch_yhat))
        return np.vstack(Yhat)
                
    def predict(self, X, batch_size=32):
        return np.argmax(self.predict_proba(X, batch_size), -1)

    def _generate_batches(self, X, batch_size=32):
        batch = []
        for sample in X:
            batch.append(sample)
            if len(batch) == batch_size:
                yield np.vstack(batch)
                batch.clear()
        if len(batch) > 0:
            yield np.vstack(batch)
        
