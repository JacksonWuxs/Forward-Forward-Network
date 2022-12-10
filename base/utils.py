"""
File: utils.py
Desc: metrics and data preprocessing functions.
"""
import numpy as np
import sklearn.metrics as metrics


def scoring(y_true, y_prob, threshold=0.5):
    """
    Inputs
    ------
    y_true : numpy.ndarray
        the target label sequence, where each element shold be 0 or 1

    y_prob : numpy.ndarray
        the same shape as `y_true` storing the predicted probability to the label `

    threshold : float (default=0.5)

    Outputs
    -------
    accuracy : float
    f1 : float
    roc_auc : float
    """
    assert y_true.shape == y_prob.shape
    assert len(y_true.shape) == len(y_prob.shape) == 1
    y_pred = np.where(y_prob >= threshold, 1.0, 0.0)
    return (metrics.accuracy_score(y_true, y_pred),
            metrics.f1_score(y_true, y_pred),
            metrics.roc_auc_score(y_true, y_prob))


def loading(root):
    data, labels = [], {}
    with open(root, encoding="utf8") as f:
        for row in f:
            if len(row) <= 1:
                break
            record = row.strip().split(",")
            record[:-1] = map(float, record[:-1])
            record[-1] = labels.setdefault(record[-1], len(labels))
            data.append(record)
    label_list = [_[0] for _ in sorted(labels.items(), key=lambda pair: pair[1])]
    return np.array(data, dtype=np.float32), label_list


def normalize(data):
    X, Y = data[:, :-1], data[:, -1:]
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-7
    return np.hstack([(X - mean) / std, Y])


def discretize(data):
    X, Y = data[:, :-1], data[:, -1:]
    mean = X.mean(axis=0)
    return np.hstack([np.where(X >= mean, 1.0, 0.0), Y])


def splitting(data, train_rate=0.7, seed=14):
    split = int(train_rate * len(data))
    np.random.seed(seed)
    np.random.shuffle(data)
    return (np.array(data[:split], dtype=np.float32),
            np.array(data[split:], dtype=np.float32))


def prepare_dataset(filepath, do_normalize=False, do_discretize=False, train_rate=0.7):
    """
    Inputs
    ------
    filepath : str
        file path to the iris dataset

    discretize : bool (default=False)
        whether we discretize continue variables to 0.0 or 1.0

    normalize : bool (default=False)
        whether we normalize continue variables with mean=0.0, std=1.0

    train_rate : float (default=0.7)
        how much percent of samples belongs to training set (others will be test)
        
    Outputs
    -------
    train : numpy.array
        numerical dataset, where last dimension
        is the label dimension.

    test : numpy.array
        numerical dataset, where last dimension
        is the label dimension.

    label_names : list
        a list of string with label names
    """
    assert 0 <= train_rate <= 1.0
    data, label_names = loading(filepath)
    if do_normalize:
        data = normalize(data)
    if do_discretize:
        data = discretize(data)    
    train, test = splitting(data)
    return (train, test, label_names)

