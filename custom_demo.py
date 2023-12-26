import argparse
import numpy as np
import torch
from pprint import pprint
import wandb

from utils import get_dataset, get_net, get_strategy, get_params
from custom_nets import Net as CustomNet, BertForNSP, SBERTCrossEncoderFinetune, SimCSECrossEncoderFinetune
from custom_data import get_STS_data


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


seed = 42
# fix random seed: https://darinabal.medium.com/deep-learning-reproducible-results-using-pytorch-42034da5ad7
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# start experiment
version = 'only_nsp_v0'
datasets = ['local_stj', 'iris_stj_local_stj', 'iris_stj']
samples = [640, 1280, 2560, 5120]
n_init_labeleds = [16, 32, 64, 128]
n_queries = [8, 16, 32, 64]
n_round = 5

# version = 'desenv_vx'
# datasets = ['local_stj']
# samples = [128]
# n_init_labeleds = [8]
# n_queries = [8]
# n_round = 2

sbert_base_models = {
    'SBERT_Local_BERTibaum': 'melll-uff/sbert_ptbr',
    'SBERT_Legal_BERTimbau': 'rufimelo/Legal-BERTimbau-sts-base-ma-v2',
    'SBERT_Paraphrase_Multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'SimCSE_LegalBERTPT-br': 'DanielJunior/legal-bert-pt-br_ulysses-camara',
    # 'SBERT_STJ_IRIS': 'stjiris/bert-large-portuguese-cased-legal-mlm-gpl-nli-sts-v1' #Estoura memoria
}
nsp_base_models = {
    'BERT': 'neuralmind/bert-base-portuguese-cased',
    'ITD_BERT': 'melll-uff/itd_bert',
    'BERTikal': 'felipemaiapolo/legalnlp-bert',
    'Legal_BERT_STF': 'dominguesm/legal-bert-base-cased-ptbr',
    # 'Legal_BERT_STJ_IRIS': 'stjiris/bert-large-portuguese-cased-legal-mlm', #Estoura memoria
    # 'Longformer': 'melll-uff/longformer', #Estoura memoria
    # 'ITD_Longformer': 'melll-uff/itd_longformer' #Estoura memoria
}
train_params = {'n_epochs': 1,
                'train_batch_size': 32 #4 para os baselines
                }

strategies = [
    "RandomSampling",
    "LeastConfidence",
    "MarginSampling",
    "EntropySampling",
    "KMeansSampling",
    "KCenterGreedy",
    "LeastConfidenceDropout",
     "MarginSamplingDropout",
     "EntropySamplingDropout",
    # "BALDDropout", #<= TODO
    # "AdversarialBIM", #<= TODO
    # "AdversarialDeepFool" #<= TODO
]

for dataset_name in datasets:
    print("============================>DATASET: " + dataset_name.upper() + "<=============================\n")
    for sample in samples:
        if dataset_name in ['iris_stj_local_stj', 'iris_stj'] and sample > 1280:
            continue
        ################################BASELINE#######################
        for model_name, model_path in sbert_base_models.items():
            print("============================>RAW BASELINE: " + model_name + "<=============================\n")
            with ClearCache():
                dataset = get_STS_data(dataset_name, sample, seed)  # load dataset

                if model_name == 'SimCSE_LegalBERTPT-br':
                    net = SimCSECrossEncoderFinetune(model_path, device)  # load network
                else:
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
                print("============================>TRAINED BASELINE: " + model_name + "<=============================\n")

                if model_name != 'SBERT_STJ_IRIS':
                    config = {"train_config": train_params,
                              "model": model_name,
                              "test_data": test_data,
                              "seed": seed,
                              "samples": sample,
                              "strategy": "TRAINED BASELINE",
                              "dataset": dataset_name.upper()}

                    run = wandb.init(project="Legal DeepAL", reinit=True, config=config, tags=[version])
                    wandb.run.name = model_name + ' -> TRAINED BASELINE'

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
                    net = None
                    dataset = None
                    print("================================================================\n")
            #############################################
        for n_init_labeled in n_init_labeleds:
            for n_query in n_queries:
                for strategy_name in strategies:
                    print("============================>" + strategy_name + "<=============================\n")
                    for model_name, model_path in nsp_base_models.items():
                        with ClearCache():
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
                            with_dropout = "Dropout" in strategy_name
                            net = CustomNet(BertForNSP, train_params, device, model_path, with_dropout)
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
                            net = None
                            dataset = None
                            print("================================================================")
