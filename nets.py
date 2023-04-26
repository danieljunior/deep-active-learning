import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample


class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.is_sbert = net.__name__ == 'SBERT_Net'
        if self.is_sbert:
            self.net = self.net(device)
        self.params = params
        self.device = device

    def train(self, data):

        if self.is_sbert:
            self.train_sbert(data, self.params)
        else:
            n_epoch = self.params['n_epoch']
            self.clf = self.net().to(self.device)
            self.clf.train()
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

            loader = DataLoader(data, shuffle=True, **self.params['train_args'])
            for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
                for batch_idx, (x, y, idxs) in enumerate(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    out, e1 = self.clf(x)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()

    def train_sbert(self, data, params):
        self.net.train(data, params)

    def predict(self, data):
        if self.is_sbert:
            return self.net.predict(data)

        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


# class SBERT_Net(nn.Module):
class SBERT_Net():
    def __init__(self, device='cpu'):
        self.model = self.build_model('models/sentence_transformer', device)

    def predict(self, data):
        embeddings_A = self.model.encode([text[0] for text in data.X], convert_to_tensor=True,
                                         show_progress_bar=False)
        embeddings_B = self.model.encode([text[1] for text in data.X], convert_to_tensor=True,
                                         show_progress_bar=False)

        # Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(embeddings_A, embeddings_B)
        return np.array([cosine_scores[i][i] for i, s in enumerate(data.X)])

    def train(self, data, options):
        train_samples = self.convert_data_to_train(data.X, data.Y)
        train_dataloader = DataLoader(train_samples, shuffle=True,
                                      batch_size=options["train_batch_size"])
        train_loss = losses.CosineSimilarityLoss(model=self.model)

        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(
            len(train_dataloader) * options["n_epochs"] * 0.1)  # 10% of train data for warm-up
        # Train the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       # evaluator=evaluator,
                       epochs=options['n_epochs'],
                       # evaluation_steps=1000,
                       warmup_steps=warmup_steps)

    def build_model(self, base_model_path, device='cpu'):
        return SentenceTransformer(base_model_path, device=device)

    def convert_data_to_train(self, X, y):
        samples = []
        for i, row in tqdm(enumerate(X)):
            score = float(y[i]) / 5.0  # Normalize score to range 0 ... 1
            example = InputExample(texts=[row[0], row[1]],
                                   label=score)
            samples.append(example)
        return samples

    def get_embeddings(self, text):
        embeddings = self.model.encode([text], convert_to_tensor=True)
        return embeddings

    def get_embedding_dim(self):
        return 50
