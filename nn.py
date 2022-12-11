import torch as tc


from base.module import GradientClassifier


class MLPBinaryClassifier(GradientClassifier):
    def __init__(self, in_dim=4, hide_dim=200, device="cpu"):
        GradientClassifier.__init__(self, "MLP")
        self._features = in_dim
        self.weights_hide = tc.nn.Parameter(tc.randn((in_dim, hide_dim)))
        self.bias_hide = tc.nn.Parameter(tc.zeros((hide_dim, ), dtype=tc.float32) + 0.1)
        self.weights_clf = tc.nn.Parameter(tc.randn((hide_dim, 1)))
        self.bias_clf = tc.nn.Parameter(tc.zeros((1, ), dtype=tc.float32) + 0.1)

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self._features
        hidden_states = tc.mm(x, self.weights_hide) + self.bias_hide
        actived_hidden_states = tc.relu(hidden_states)
        logits = tc.mm(actived_hidden_states, self.weights_clf) + self.bias_clf
        return tc.sigmoid(logits)
