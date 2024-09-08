import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import warnings
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*gamma.*")
warnings.filterwarnings("ignore", message=".*beta.*")

class EssayDataset(Dataset):
    def __init__(self, essays, labels, tokenizer, max_len):
        self.essays = essays
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, item):
        essay = str(self.essays[item])
        labels = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'essay_text': essay,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }
    
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = EssayDataset(
        essays=df[['Essay']].to_numpy(),
        labels=df[['Grammar Score', 'Vocabulary Score', 'Coherence Score', 'Naturalness Score']].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


df = pd.read_csv('generated_essays_500.csv')


train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)

MAX_LEN = 128
BATCH_SIZE = 8

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
loss_fn = torch.nn.CrossEntropyLoss().to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

import torch.nn as nn

class EssayScoringModel(nn.Module):
    def __init__(self, n_outputs):
        super(EssayScoringModel, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_outputs)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(bert_output.pooler_output)
        return self.out(output)
    
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    preds = []
    true_labels = []

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.append(outputs.detach().cpu().numpy())
        true_labels.append(labels.detach().cpu().numpy())

    return np.mean(losses), np.vstack(preds), np.vstack(true_labels)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    preds = []
    true_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            preds.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    return np.mean(losses), np.vstack(preds), np.vstack(true_labels)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = EssayScoringModel(n_outputs=4)
    model = model.to(device)

    EPOCHS = 10
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    loss_fn = torch.nn.MSELoss().to(device)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_loss, train_preds, train_labels = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(train_df)
        )

        print(f'Train loss {train_loss}')

        val_loss, val_preds, val_labels = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(test_df)
        )

        print(f'Val loss {val_loss}')

    print(val_preds)

    kappas = []
    for i in range(4):
        preds = np.round(val_preds[:, i])
        true = np.round(val_labels[:, i])
        kappa = cohen_kappa_score(true, preds, weights='quadratic')
        kappas.append(kappa)
        print(f'Kappa for feature {i+1}: {kappa}')

    weighted_kappa = np.average(kappas, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Weighted Kappa: {weighted_kappa}')