import argparse
import numpy as np
import torch
from pprint import pprint
import wandb

from utils import get_dataset, get_net, get_strategy, get_params
from custom_nets import Net as CustomNet, BertForNSP
from custom_data import get_STS_data
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
version = 'dgx_v1'
seeds = [10]
datasets = ['local_stj', 'iris_stj_local_stj', 'iris_stj']
samples = [250, 500, 1000, 2000]
n_init_labeleds = [10, 20, 50, 100]
n_queries = [10, 20, 30]
n_round = 5

# version = 'desenv_v0'
# seeds = [10]
# datasets = ['local_stj', 'iris_stj_local_stj', 'iris_stj']
# # datasets = ['local_stj']
# samples = [50]
# n_init_labeleds = [10]
# n_queries = [10]
# n_round = 2

sbert_base_models = {
    'SBERT_BERTibaum': 'melll-uff/sbert_ptbr',
    'SimCSE_LegalBERTPT-br': 'DanielJunior/legal-bert-pt-br_ulysses-camara',
    'SBERT_STJ_IRIS': 'stjiris/bert-large-portuguese-cased-legal-mlm-gpl-nli-sts-v1'
    }
nsp_base_models = {
    'BERT': 'neuralmind/bert-base-portuguese-cased',
    'ITD_BERT': 'melll-uff/itd_bert',
    'BERTikal': 'felipemaiapolo/legalnlp-bert',
    'Legal_BERT_STJ_IRIS': 'stjiris/bert-large-portuguese-cased-legal-mlm',
    'Legal_BERT_STF': 'dominguesm/legal-bert-base-cased-ptbr',
    'Longformer': 'melll-uff/longformer',
    'ITD_Longformer': 'melll-uff/itd_longformer'
    }
train_params = {'n_epochs': 1,
                # 'train_batch_size': 4
                'train_batch_size': 16
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
for dataset_name in datasets:
    print("============================>DATASET: " + dataset_name.upper() + "<=============================\n")
    for seed in seeds:
        for sample in samples:
            ################################BASELINE#######################
            print("============================>BASELINE<=============================\n")
            for model_name, model_path in sbert_base_models.items():

                dataset = get_STS_data(dataset_name, sample, seed)  # load dataset
                net = SBERTCrossEncoderFinetune(model_path, device)  # load network
                test_data = np.fromiter(map(lambda x: x.label, dataset.get_test_data()), dtype=int)

                config = {"train_config": train_params,
                          "model": model_name,
                          "test_data": test_data,
                          "seed": seed,
                          "samples": sample,
                          "strategy": "RAW BASELINE",
                          "dataset": dataset_name.upper()}

                run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
                wandb.run.name = model_name + ' -> RAW BASELINE'

                preds = net.predict(dataset.get_test_data()).cpu()
                accuracy = dataset.cal_test_acc(preds)
                print(f"Raw Baseline testing accuracy: {accuracy}")

                wandb.log({
                    "total_labeled_data": 0,
                    'predictions': preds.numpy(),
                    'test_accuracy': accuracy})
                run.finish()
                print("================================================================\n")

                if model_name != 'SBERT_STJ_IRIS':
                    config = {"train_config": train_params,
                              "model": model_name,
                              "test_data": test_data,
                              "seed": seed,
                              "samples": sample,
                              "strategy": "ALL TRAIN DATA",
                              "dataset": dataset_name.upper()}

                    run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
                    wandb.run.name = model_name + ' -> ALL TRAIN BASELINE'

                    _, all_train_data = dataset.get_train_data()
                    net.train(all_train_data, train_params)
                    preds = net.predict(dataset.get_test_data()).cpu()
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
                        print("============================>" + strategy_name + "<=============================\n")
                        for model_name, model_path in nsp_base_models.items():
                            print("============================>" + model_name + "<=============================\n")
                            config = {
                                "n_init_labeled": n_init_labeled,
                                "n_round": n_round,
                                "n_query": n_query,
                                "train_config": train_params,
                                "model": model_name,
                                "seed": seed,
                                "samples": sample,
                                "dataset": dataset_name.upper()
                            }

                            dataset = get_STS_data(dataset_name, sample, seed)  # load dataset
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
                            preds = strategy.predict(dataset.get_test_data()).cpu()
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
                                preds = strategy.predict(dataset.get_test_data()).cpu()
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
