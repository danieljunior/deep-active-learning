import argparse
import numpy as np
import torch
from pprint import pprint
import wandb

from utils import get_dataset, get_net, get_strategy, get_params
from handlers import STS_Handler
from custom_nets import Net as CustomNet, BertForNSP
from custom_data import get_STS_Classification
from nets import SBERTCrossEncoderFinetune

seed = 42
# fix random seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# start experiment
version = 'v6'
seeds = [10]
samples = [250, 500, 1000]
n_init_labeleds = [10, 20, 50]
n_queries = [10, 20, 30]
n_round = 5

# seeds = [10]
# samples = [50]
# n_init_labeleds = [10]
# n_queries = [10]
# n_round = 2

base_models = {
    'BERT': 'models/bert-base-cased-pt-br',
    'BERTikal': 'models/BERTikal',
    'ITD_BERT': 'models/itd_bert',
    'Longformer': 'models/bert_longformer',
    'ITD_Longformer': 'models/itd_bert_longformer'
}

train_params = {'n_epochs': 1,
                'train_batch_size': 4
                }

strategies = [
    "RandomSampling",
    "LeastConfidence",
    "MarginSampling",
    "EntropySampling",
    "KMeansSampling",
    "KCenterGreedy",
    # "LeastConfidenceDropout", #<= TODO
    # "MarginSamplingDropout", #<= TODO
    # "EntropySamplingDropout", #<= TODO
    # "BALDDropout", #<= TODO
    # "AdversarialBIM", #<= TODO
    # "AdversarialDeepFool" #<= TODO
]
for seed in seeds:
    for sample in samples:
        ################################BASELINE#######################
        print("============================>BASELINE<=============================\n")
        baseline_model = 'SBERTCrossEncoderFinetune'
        dataset = get_dataset(baseline_model, sample, seed)  # load dataset
        net = SBERTCrossEncoderFinetune(device)  # load network

        _, all_train_data = dataset.get_train_data()
        test_data = np.fromiter(map(lambda x: x.label, dataset.get_test_data()), dtype=int)

        config = {"train_config": train_params,
                  "model": baseline_model,
                  "test_data": test_data,
                  "seed": seed,
                  "samples": sample,
                  "strategy": "ALL TRAIN DATA"}

        run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
        wandb.run.name = baseline_model + ' -> BASELINE'

        net.train(all_train_data, train_params)
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
        for n_init_labeled in n_init_labeleds:
            for n_query in n_queries:
                for strategy_name in strategies:
                    print("==========>" + strategy_name + "<===========\n")
                    for model_name, model_path in base_models.items():
                        print("==========>" + model_name + "<===========\n")

                        config = {
                            "n_init_labeled": n_init_labeled,
                            "n_round": n_round,
                            "n_query": n_query,
                            "train_config": train_params,
                            "model": model_name,
                            "seed": seed,
                            "samples": sample
                        }

                        dataset = get_STS_Classification(STS_Handler, sample, seed)
                        net = CustomNet(BertForNSP, train_params, device, model_path)
                        test_data = np.fromiter(map(lambda x: x.label, dataset.get_test_data()), dtype=int)

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

                        for rd in range(1, n_round + 1):
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
                        run.finish()
                    print("================================================================")
