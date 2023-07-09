import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy, get_params
from pprint import pprint

from nets import SBERTCrossEncoderFinetune
import wandb

seed = 42
# fix random seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

version = 'v2'

##############BASELINE#######################
print("============================>BASELINE<=============================\n")
baseline_model = 'SBERTCrossEncoderFinetune'
dataset = get_dataset(baseline_model)  # load dataset
net = SBERTCrossEncoderFinetune(device)  # load network

params = {'n_epochs': 1,
          'train_batch_size': 4}

_, all_train_data = dataset.get_train_data()
test_data = np.fromiter(map(lambda x: x.label, dataset.get_test_data()), dtype=int)

config = {"train_config": params,
          "model": baseline_model,
          "test_data": test_data,
          "strategy": "ALL TRAIN DATA"}

run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
wandb.run.name = baseline_model + ' -> BASELINE'

net.train(all_train_data, params)
# calculate accuracy
preds = net.predict(dataset.get_test_data())
accuracy = dataset.cal_test_acc(preds)

print(f"Baseline testing accuracy: {accuracy}")

wandb.log({
    "total_labeled_data": len(all_train_data),
    'predictions': preds.numpy(),
    'test_accuracy': accuracy})
run.finish()
print("================================================================\n")
#############################################

# start experiment
n_init_labeled = 12
n_round = 5
n_query = 8
model_name = 'SBERTCrossEncoderFinetune'

strategies = [
    "RandomSampling",
    "LeastConfidence",
    "MarginSampling",
    "EntropySampling",
    # "LeastConfidenceDropout", #<= TODO
    # "MarginSamplingDropout", #<= TODO
    # "EntropySamplingDropout", #<= TODO
    # "KMeansSampling", #<= TODO
    # "KCenterGreedy", #<= TODO
    # "BALDDropout", #<= TODO
    # "AdversarialBIM", #<= TODO
    # "AdversarialDeepFool" #<= TODO
]

config = {
    "n_init_labeled": n_init_labeled,
    "n_round": n_round,
    "n_query": n_query,
    "train_config": get_params(model_name),
    "model": model_name
}

for strategy_name in strategies:
    dataset = get_dataset(model_name)  # load dataset
    net = get_net(model_name, device)  # load network
    test_data = np.fromiter(map(lambda x: x.label, dataset.get_test_data()), dtype=int)

    print("==========>" + strategy_name + "<===========\n")
    strategy = get_strategy(strategy_name)(dataset, net)  # load strategy

    dataset.initialize_labels(n_init_labeled)
    print(f"number of labeled pool: {n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool - n_init_labeled}")
    print(f"number of unlabeled query data: {n_query}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # round 0 accuracy
    print("Round 0")
    _, initial_labeled_data = dataset.get_labeled_data()
    initial_labeled_data = np.fromiter(map(lambda x: x.label, initial_labeled_data), dtype=int)
    print(f"Initial labeled data: \t{initial_labeled_data}")

    config["n_initial_unlabeled_pool"] = dataset.n_pool - n_init_labeled
    config["initial_labeled_data"] = initial_labeled_data,
    config['test_data'] = test_data
    config['strategy'] = strategy_name

    run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
    # wandb.run.name = model_name + ' -> ' + strategy_name + ' -> Round 0'
    wandb.run.name = model_name + ' -> ' + strategy_name

    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    accuracy = dataset.cal_test_acc(preds)
    print(f"Round 0 testing accuracy: {accuracy}")
    print(f"Test data: \t{test_data}")
    print(f"Predictions: \t{preds.numpy()}")

    wandb.log({
        "round": 0,
        "new_labeled_data": initial_labeled_data,
        "predictions": preds.numpy(),
        "test_accuracy": accuracy,
        "total_labeled_data": len(initial_labeled_data)})
    # run.finish()

    for rd in range(1, n_round + 1):
        # run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
        # wandb.run.name = model_name + ' -> ' + strategy_name + " -> Round " + str(rd)
        print(f"\nRound {rd}")

        # query
        query_idxs = strategy.query(n_query)
        strategy.update(query_idxs)
        _, new_labeled_data = dataset.get_query_data(query_idxs)
        _, labeled_data = dataset.get_labeled_data()
        new_labeled_data = np.fromiter(map(lambda x: x.label, new_labeled_data), dtype=int)
        print(f"New labeled data: \t{new_labeled_data}")
        # update labels
        # strategy.train()
        strategy.incremental_train(query_idxs)

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        accuracy = dataset.cal_test_acc(preds)

        print(f"Round {rd} testing accuracy: {accuracy}")
        print(f"Test data: \t{test_data}")
        print(f"Predictions: \t{preds.numpy()}")

        wandb.log({
            "round": rd,
            "new_labeled_data": new_labeled_data,
            "predictions": preds.numpy(),
            "total_labeled_data": len(labeled_data),
            'test_accuracy': accuracy
        })
        # run.finish()
    run.finish()
    print("================================================================")
