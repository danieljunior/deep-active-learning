import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForNextSentencePrediction
import numpy as np
from tqdm import tqdm  # for our progress bar
from utils import get_dataset


class STSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

baseline_model = 'BertClassification'
raw_dataset = get_dataset(baseline_model)

_, all_train_data = raw_dataset.get_train_data()
documents_pair = np.array(list(map(lambda x: x.texts, all_train_data)))
labels = np.fromiter(map(lambda x: x.label, all_train_data), dtype=int).tolist()

tokenizer = BertTokenizer.from_pretrained('models/bert-base-cased-pt-br')
model = BertForNextSentencePrediction.from_pretrained('models/bert-base-cased-pt-br')

inputs = tokenizer(documents_pair[:, 0].tolist(), documents_pair[:, 1].tolist(),
                   return_tensors='pt', max_length=512, truncation=True, padding='max_length')
inputs['labels'] = inputs['labels'] = torch.LongTensor([labels]).T

dataset = STSDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# activate training mode
model.train()
# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=5e-6)

epochs = 1

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model.eval()
all_test_data = raw_dataset.get_test_data()
test_documents_pair = np.array(list(map(lambda x: x.texts, all_test_data)))
test_labels = np.fromiter(map(lambda x: x.label, all_test_data), dtype=int).tolist()
test_inputs = tokenizer(test_documents_pair[:, 0].tolist(), test_documents_pair[:, 1].tolist(),
                        return_tensors='pt', max_length=512, truncation=True, padding='max_length')
test_inputs['labels'] = torch.LongTensor([test_labels]).T
test_dataset = STSDataset(test_inputs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
loop = tqdm(test_loader, leave=True)

true_labels = []
predicted_labels = []
predicted_probs = []
for batch in loop:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    true_labels.append(batch['labels'].T[0])
    # process
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    # labels=labels,
                    output_hidden_states=True)
    predicted_labels.append(outputs.logits.data.max(1)[1])
    predicted_probs.append(torch.nn.functional.softmax(outputs.logits.data, dim=1))
    # to get embbedings
    # hidden_states = outputs.hidden_states
    # concatened_last_four_layers = torch.cat(outputs.hidden_states[-4:], -1) #tensor of shape (batch_size, seq_len, 4 * hidden_size)

test_acc = 1.0 * (torch.cat(true_labels) == torch.cat(predicted_labels)).sum().item() / len(torch.cat(true_labels))
print('Test accuracy:', test_acc)