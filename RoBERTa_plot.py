import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import warnings
import logging
import matplotlib.pyplot as plt


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

class EssayScoringModel(torch.nn.Module):
    def __init__(self, n_outputs, model_name):
        super(EssayScoringModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.out = torch.nn.Linear(self.roberta.config.hidden_size, n_outputs)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = self.dropout(roberta_output.last_hidden_state[:, 0, :])  # Use the [CLS] token representation
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


def train_and_evaluate(train_df, test_df, tokenizer, max_len, batch_size, epochs, learning_rate):
    train_data_loader = create_data_loader(train_df, tokenizer, max_len, batch_size)
    test_data_loader = create_data_loader(test_df, tokenizer, max_len, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EssayScoringModel(n_outputs=4, model_name=PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    loss_fn = torch.nn.MSELoss().to(device)

    best_val_loss = float('inf')
    epoch_scores = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        train_loss, _, _ = train_epoch(
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

        kappas = []
        for i in range(4):
            preds = np.round(val_preds[:, i])
            true = np.round(val_labels[:, i])
            kappa = cohen_kappa_score(true, preds, weights='quadratic')
            kappas.append(kappa)

        weighted_kappa = np.average(kappas, weights=[0.25, 0.25, 0.25, 0.25])
        epoch_scores.append(kappas + [weighted_kappa])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    _, val_preds, val_labels = eval_model(model, test_data_loader, loss_fn, device, len(test_df))

    final_kappas = []
    for i in range(4):
        preds = np.round(val_preds[:, i])
        true = np.round(val_labels[:, i])
        kappa = cohen_kappa_score(true, preds, weights='quadratic')
        final_kappas.append(kappa)
        print(f'Kappa for feature {i+1}: {kappa}')

    final_weighted_kappa = np.average(final_kappas, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Weighted Kappa: {final_weighted_kappa}')
    
    return final_kappas + [final_weighted_kappa], epoch_scores

if __name__ == "__main__":
    df = pd.read_csv('generated_essays_500.csv')
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe

    PRE_TRAINED_MODEL_NAME = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    MAX_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-5

    essay_counts = list(range(50, len(df) + 1, 50))  # Start from 50, increment by 50
    results = []
    all_epoch_scores = {}

    for count in essay_counts:
        print(f"\nTraining with {count} essays")
        subset_df = df.iloc[:count]
        train_df, test_df = train_test_split(subset_df, test_size=0.3)

        scores, epoch_scores = train_and_evaluate(train_df, test_df, tokenizer, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE)
        results.append(scores)
        all_epoch_scores[count] = epoch_scores

    results = np.array(results)

    plt.figure(figsize=(12, 8))
    metrics = ['Grammar', 'Vocabulary', 'Coherence', 'Naturalness', 'Weighted Kappa']
    colors = ['r', 'g', 'b', 'c', 'm']
    
    for i, metric in enumerate(metrics):
        plt.plot(essay_counts, results[:, i], marker='o', label=metric, color=colors[i])
    
    plt.title('Model Performance vs Number of Essays')
    plt.xlabel('Number of Essays')
    plt.ylabel('Kappa Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('performances_vs_data_size.png')
    plt.close()

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.ravel()

    for i, metric in enumerate(metrics):
        for count in essay_counts:
            epoch_scores = np.array(all_epoch_scores[count])
            axs[i].plot(range(1, EPOCHS + 1), epoch_scores[:, i], marker='o', label=f'{count} essays')
        
        axs[i].set_title(f'{metric} Score vs Epochs')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Kappa Score')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('validation_scores_vs_epochs.png')
    plt.close()

    print("\nResults:")
    for count, scores in zip(essay_counts, results):
        print(f"Number of essays: {count}")
        for metric, score in zip(metrics, scores):
            print(f"  {metric}: {score:.4f}")
        print()

    for i, metric in enumerate(metrics):
        best_count = essay_counts[np.argmax(results[:, i])]
        best_score = np.max(results[:, i])
        print(f"Best {metric} score: {best_score:.4f} with {best_count} essays")