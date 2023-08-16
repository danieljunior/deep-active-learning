from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler, STS_Handler
from custom_data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10, get_STJ_STS, \
    get_STJ_STS_Classification, get_STS_Classification
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, SBERT_Net, SBERT_CrossEncoder, \
    SBERTCrossEncoderFinetune
from custom_nets import BertForNSP
from custom_nets import Net as CustomNet
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, \
    AdversarialBIM, AdversarialDeepFool

params = {'MNIST':
              {'n_epoch': 10,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'FashionMNIST':
              {'n_epoch': 10,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'SVHN':
              {'n_epoch': 20,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'CIFAR10':
              {'n_epoch': 20,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.05, 'momentum': 0.3}},
          'STS':
              {'n_epochs': 2,
               'train_batch_size': 8
               },
          'STS_Cross':
              {'n_epochs': 2,
               'train_batch_size': 4
               },
          'SBERTCrossEncoderFinetune':
              {'n_epochs': 1,
               'train_batch_size': 4
               },
          'BertClassification':
              {'n_epochs': 1,
               'train_batch_size': 4
               }
          }


def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return MNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    elif name in ['STS', 'STS_Classification', 'SBERTCrossEncoderFinetune', 'BertClassification']:
        return STS_Handler


def get_dataset(name, sample=10, seed=42):
    if name == 'MNIST':
        return get_MNIST(get_handler(name))
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(name))
    elif name == 'SVHN':
        return get_SVHN(get_handler(name))
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name))
    elif name == 'STS':
        return get_STJ_STS(get_handler(name))
    elif name == 'STS_Classification':
        return get_STJ_STS_Classification(get_handler(name))
    elif name in ['SBERTCrossEncoderFinetune', 'BertClassification']:
            return get_STS_Classification(get_handler(name), sample, seed)
    else:
        raise NotImplementedError


def get_net(name, device):
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device)
    elif name == 'STS':
        return Net(SBERT_Net, params[name], device)
    elif name == 'STS_Cross':
        return Net(SBERT_CrossEncoder, params[name], device)
    elif name == 'SBERTCrossEncoderFinetune':
        return Net(SBERTCrossEncoderFinetune, params[name], device)
    elif name == 'BertClassification':
        return CustomNet(BertForNSP, params[name], device)
    else:
        raise NotImplementedError


def get_params(name):
    return params[name]


def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "BALD":
        return BALD
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError

# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
