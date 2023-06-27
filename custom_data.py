import numpy as np
import torch
from torchvision import datasets
import pandas as pd
from scipy.stats import pearsonr
from sentence_transformers.readers import InputExample


class CustomData:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.n_pool = len(train)
        self.n_test = len(test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.train[labeled_idxs]

    def get_query_data(self, query_idxs):
        labeled_idxs = np.arange(self.n_pool)[query_idxs]
        return labeled_idxs, self.train[labeled_idxs]

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.train[unlabeled_idxs]

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.train

    def get_test_data(self):
        return self.test

    def cal_test_acc(self, preds):
        y_test = torch.from_numpy(np.fromiter(map(lambda x: x.label, self.test), dtype=int))
        return 1.0 * (y_test == preds).sum().item() / self.n_test


def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000],
                handler)


def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000],
                handler)


def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000],
                torch.from_numpy(data_test.labels)[:40000], handler)


def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000],
                torch.LongTensor(data_test.targets)[:40000], handler)


def get_STJ_STS(handler):
    data_train = pd.read_csv('./data/STS/train.csv')
    data_test = pd.read_csv('./data/STS/test.csv')
    X_train = data_train[['sentence_A', 'sentence_B']].values
    y_train = data_train.score.values
    X_test = data_test[['sentence_A', 'sentence_B']].values
    y_test = data_test.score.values
    return Data(X_train[:1000], torch.LongTensor(y_train[:1000]),
                X_test, torch.LongTensor(y_test), handler)


def get_STJ_STS_Classification(handler):
    data_train = pd.read_csv('./data/STS/train.csv').sample(n=1000, random_state=42)
    data_test = pd.read_csv('./data/STS/test.csv')
    X_train = data_train[['sentence_A', 'sentence_B']].values
    y_train = np.fromiter(map(lambda x: 1 if x >= 4 else 0, data_train.score.values), dtype=int)
    X_test = data_test[['sentence_A', 'sentence_B']].values
    y_test = np.fromiter(map(lambda x: 1 if x >= 4 else 0, data_test.score.values), dtype=int)
    return Data(X_train, torch.LongTensor(y_train),
                X_test, torch.LongTensor(y_test), handler)


def get_STS_Classification(handler):
    data_train = pd.read_csv('./data/STS/train.csv').sample(n=100, random_state=10)
    X_train = data_train[['sentence_A', 'sentence_B']].values
    y_train = np.fromiter(map(lambda x: 1 if x >= 4 else 0, data_train.score.values), dtype=int)
    train = []
    for i, row in enumerate(X_train):
        example = InputExample(texts=[row[0], row[1]], label=y_train[i])
        train.append(example)

    data_test = pd.read_csv('./data/STS/test.csv')
    X_test = data_test[['sentence_A', 'sentence_B']].values
    y_test = np.fromiter(map(lambda x: 1 if x >= 4 else 0, data_test.score.values), dtype=int)
    test = []
    for i, row in enumerate(X_test):
        example = InputExample(texts=[row[0], row[1]], label=y_test[i])
        test.append(example)
    return CustomData(np.array(train), np.array(test))
