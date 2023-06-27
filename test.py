import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint

seed = 42
# fix random seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset('SBERTCrossEncoderFinetune')                   # load dataset
# dataset = get_dataset('STS')                   # load dataset
net = get_net('SBERTCrossEncoderFinetune', device)                   # load network
# net = get_net('STS', device)                   # load network
# strategy = get_strategy('RandomSampling')(dataset, net)  # load strategy
strategy = get_strategy('LeastConfidence')(dataset, net)  # load strategy

# start experiment
n_init_labeled = 10
n_round = 8
n_query = 12
dataset.initialize_labels(n_init_labeled)
print(f"number of labeled pool: {n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())


print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")
print(preds)
for rd in range(1, n_round+1):
    print("#############################################")
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(n_query)

    # update labels
    strategy.update(query_idxs)
    # strategy.train()
    strategy.incremental_train(query_idxs)

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())
    print(preds)
    # print(f"Round {rd} testing accuracy: {dataset.pearson_correlation(preds)}")
    print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
