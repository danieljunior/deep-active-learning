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
datasets = ['local_stj']
n_round = 5

setups = [
    {2560: [
        {'KCenterGreedy': [
            {'il': 64, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]},
        {'KMeansSampling': [
            {'il': 64, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]},
        {'LeastConfidence': [
            {'il': 32, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]},
        {'MarginSampling': [
            {'il': 32, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]},
        {'RandomSampling': [
            {'il': 32, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 64, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 128, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]}
    ]},

    {5120: [
        {'LeastConfidenceDropout': [
            {'il': 16, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 16, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 16, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 16, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]},
        {'MarginSamplingDropout': [
            {'il': 16, 'q': 8, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 16, 'q': 16, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 16, 'q': 32, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
            {'il': 16, 'q': 64, 'models': ['ITD_BERT', 'Legal_BERT_STF', 'BERT', 'BERTikal']},
        ]},
    ]}
]

nsp_base_models = {
    'BERT': 'neuralmind/bert-base-portuguese-cased',
    'ITD_BERT': 'melll-uff/itd_bert',
    'BERTikal': 'felipemaiapolo/legalnlp-bert',
    'Legal_BERT_STF': 'dominguesm/legal-bert-base-cased-ptbr',
}
train_params = {'n_epochs': 1,
                'train_batch_size': 32  # 4 para os baselines
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
    for setup in setups:
        for sample, v1 in setup.items():
            for al_setups in v1:
                for strategy_name, v2 in al_setups.items():
                    print("============================>" + strategy_name + "<=============================\n")
                    for params in v2:
                        n_init_labeled = params['il']
                        n_query = params['q']
                        for model_name in params['models']:
                            print("============================>" + model_name + "<=============================\n")
                            model_path = nsp_base_models[model_name]

                            with ClearCache():
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

                                # round 0 accuracy ### DAQUI
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
