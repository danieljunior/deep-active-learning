import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForNextSentencePrediction, AutoConfig
from transformers import TrainingArguments, Trainer, logging

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator

from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.set_verbosity_error()


class Net:
    def __init__(self, net, params, device, model_path='models/bert-base-cased-pt-br', with_dropout=False):
        self.net = net(device, model_path, with_dropout)
        self.params = params
        self.device = device

    def train(self, data):
        self.net.train(data, self.params)

    def predict(self, data):
        return self.net.predict(data)

    def predict_prob(self, data):
        return self.net.predict_prob(data)

    def predict_prob_dropout(self, data, n_drop=10):
        return self.net.predict_prob_dropout(data, n_drop)

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
        return self.net.get_embeddings(data)


class SBERT_Net:
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

    def predict_prob(self, data):
        pass

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
        for i, row in tqdm(enumerate(X), desc='Converting training data: '):
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


class SBERT_CrossEncoder:
    PREDICT_BATCH_SIZE = 16

    def __init__(self, device='cpu'):
        self.model = None
        self.device = device

    def predict(self, data):
        return self.model.predict(data.X, apply_softmax=True, batch_size=SBERT_CrossEncoder.PREDICT_BATCH_SIZE,
                                  convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=True)

    def predict_prob(self, data):
        return self.model.predict(data.X, apply_softmax=True, batch_size=SBERT_CrossEncoder.PREDICT_BATCH_SIZE,
                                  convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=True)

    def train(self, data, options):
        # TODO implementar com continuação de treinamento
        # Precisa alterar a forma como a estratégia retorna os novos exemplos
        # Hoje o data é data Tn-1 + novos exemplos, e não só novos exemplos
        self.model = self.build_model('models/bert-base-cased-pt-br', self.device)
        samples = self.convert_data_to_train(data.X, data.Y)
        validation_percent = 0.25
        validation_samples = samples[:int((validation_percent * len(samples)))]
        train_samples = samples[int((validation_percent * len(samples))):]
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=options["train_batch_size"])
        # evaluator = CEBinaryClassificationEvaluator.from_input_examples(train_samples, name='sts-dev')

        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(validation_samples, name='sts-dev')

        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(
            len(train_dataloader) * options["n_epochs"] * 0.1)  # 10% of train data for warm-up
        # Train the model
        self.model.fit(train_dataloader=train_dataloader,
                       evaluator=evaluator,
                       epochs=options["n_epochs"],
                       warmup_steps=warmup_steps,
                       show_progress_bar=True,
                       evaluation_steps=20)

    def build_model(self, base_model_path, device='cpu'):
        return CrossEncoder(base_model_path, num_labels=2, max_length=512, device=device)

    def convert_data_to_train(self, X, y):
        samples = []
        for i, row in tqdm(enumerate(X), desc='Converting training data: '):
            example = InputExample(texts=[row[0], row[1]], label=y[i])
            samples.append(example)
        return samples

    def get_embeddings(self, text):
        # embeddings = self.model.encode([text], convert_to_tensor=True)
        # return embeddings
        raise 'Not yet implemented'

    def get_embedding_dim(self):
        # return 50
        raise 'Not yet implemented'


class SBERTCrossEncoderFinetune:
    PREDICT_BATCH_SIZE = 16

    def __init__(self, model_path, device='cpu'):
        self.model = self.build_model(model_path, device)

    def predict(self, data):
        return self.model.predict(self.predict_data(data), apply_softmax=True,
                                  batch_size=SBERTCrossEncoderFinetune.PREDICT_BATCH_SIZE,
                                  convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=True).argmax(axis=1)

    def predict_prob(self, data):
        return self.model.predict(self.predict_data(data), apply_softmax=True,
                                  batch_size=SBERTCrossEncoderFinetune.PREDICT_BATCH_SIZE,
                                  convert_to_tensor=True, convert_to_numpy=False, show_progress_bar=True)

    def predict_data(self, data):
        return np.array([x.texts for x in data])

    def train(self, data, options):
        train_dataloader = DataLoader(data, shuffle=True, batch_size=options["train_batch_size"])
        # Configure the training. We skip evaluation in this example
        warmup_steps = math.ceil(
            len(train_dataloader) * options["n_epochs"] * 0.1)  # 10% of train data for warm-up
        # Train the model
        self.model.fit(train_dataloader=train_dataloader,
                       epochs=options["n_epochs"],
                       warmup_steps=warmup_steps,
                       show_progress_bar=True,
                       )

    def build_model(self, base_model_path, device='cpu'):
        return CrossEncoder(base_model_path, num_labels=2, max_length=512, device=device)


class SimCSECrossEncoderFinetune:
    PREDICT_BATCH_SIZE = 16

    def __init__(self, model_path, device='cpu'):
        word_embedding_model = models.Transformer(model_path, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)

    def predict(self, data):
        embeddings_A = self.model.encode([x.texts[0] for x in data], convert_to_tensor=True,
                                         show_progress_bar=False)
        embeddings_B = self.model.encode([x.texts[1] for x in data], convert_to_tensor=True,
                                         show_progress_bar=False)

        # Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(embeddings_A, embeddings_B).cpu()
        resp = np.array([cosine_scores[i][i] for i, s in enumerate(data)])
        return torch.from_numpy(np.array([round(e) for e in resp]))

    def predict_prob(self, data):
        raise 'Not supported'

    def predict_data(self, data):
        return np.array([x.texts for x in data])

    # https://www.sbert.net/examples/unsupervised_learning/SimCSE/README.html
    def train(self, data, options):
        train_data = self.convert_data(data)

        # DataLoader to batch your data
        train_dataloader = DataLoader(train_data, batch_size=options["train_batch_size"], shuffle=True)

        # Use the denoising auto-encoder loss
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Call the fit method
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=options["n_epochs"],
            show_progress_bar=True
        )

    def convert_data(self, data):
        return [InputExample(texts=[s, s]) for s in np.unique(np.array([pair.texts for pair in data]).flatten())]


class BertForNSP:
    def __init__(self, device='cpu', model_path='models/bert-base-cased-pt-br', with_dropout=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 4096 if 'longformer' in model_path else 512
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        if with_dropout:
            configuration = AutoConfig.from_pretrained(model_path)
            configuration.hidden_dropout_prob = 0.5
            configuration.attention_probs_dropout_prob = 0.5
            self.model = BertForNextSentencePrediction.from_pretrained(pretrained_model_name_or_path=model_path,
                                                   config=configuration)

        else:
            self.model = BertForNextSentencePrediction.from_pretrained(model_path)

        self.model.to(self.device)

    def predict_outputs(self, batch, to_train=False):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        token_type_ids = batch['token_type_ids'].to(self.device)
        if to_train:
            labels = batch['labels'].to(self.device)
            return self.model(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              labels=labels)
        else:
            return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              output_hidden_states=True)

    def predict(self, data, output_probs=False, with_dropout=False):
        test_loader = self.convert_data_to_train(data, shuffle=not with_dropout)
        loop = tqdm(test_loader, leave=True)

        true_labels = []
        predicted_labels = []
        predicted_probs = []

        if with_dropout:
            self.model.train()
        else:
            self.model.eval()

        for batch in loop:
            outputs = self.predict_outputs(batch)
            true_labels.append(batch['labels'].T[0])
            predicted_labels.append(outputs.logits.data.max(1)[1])
            predicted_probs.append(torch.nn.functional.softmax(outputs.logits.data, dim=1))

        if output_probs:
            return torch.cat(predicted_labels), torch.cat(predicted_probs)
        else:
            return torch.cat(predicted_labels)

    def predict_prob(self, data):
        _, probs = self.predict(data, output_probs=True)
        return probs

    #https://stackoverflow.com/a/71827094
    def predict_prob_dropout(self, data, n_drop=10):
        probs = torch.zeros([len(data), 2]).to(self.device)
        for i in range(n_drop):
            with torch.no_grad():
                _, probs_ = self.predict(data, output_probs=True)
                probs += probs_
        probs /= n_drop
        return probs

    def train(self, data, options):
        train_dataloader = self.convert_data_to_train(data, options)
        default_args = {
            "output_dir": "tmp",
            "evaluation_strategy": "steps",
            "num_train_epochs": options["n_epochs"],
            "log_level": "error",
            "report_to": "none",
        }
        training_args = TrainingArguments(per_device_train_batch_size=1,
                                          gradient_accumulation_steps=options["train_batch_size"],
                                          gradient_checkpointing=True,
                                          fp16=torch.cuda.is_available(),
                                          **default_args)

        trainer = Trainer(model=self.model, args=training_args, train_dataset=train_dataloader.dataset)
        result = trainer.train()
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

    def convert_data_to_train(self, data, options={"train_batch_size": 4}, shuffle=True):
        X = np.array(list(map(lambda x: x.texts, data)))
        y = np.fromiter(map(lambda x: x.label, data), dtype=int).tolist()

        inputs = self.tokenizer(X[:, 0].tolist(), X[:, 1].tolist(),
                                return_tensors='pt', max_length=self.max_length, truncation=True,
                                padding='max_length')
        inputs['labels'] = inputs['labels'] = torch.LongTensor([y]).T
        dataset = self.STSDataset(inputs)
        return torch.utils.data.DataLoader(dataset, batch_size=options["train_batch_size"], shuffle=shuffle)

    def get_embeddings(self, data):
        dataloader = self.convert_data_to_train(data)
        loop = tqdm(dataloader, leave=True)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for batch in loop:
                batch_output = self.predict_outputs(batch)
                pooled_output = torch.cat(tuple([batch_output.hidden_states[i]
                                                 for i in [-4, -3, -2, -1]]),
                                          dim=-1)
                outputs.append(pooled_output[:, 0, :].cpu())
        return torch.cat(outputs)

    def get_embedding_dim(self):
        return 3072

    class STSDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)
