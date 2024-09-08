import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import warnings
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv
from collections import Counter
import random
import anthropic

load_dotenv()

client = OpenAI(api_key='OPENAI_API_KEY')
clientC = anthropic.Anthropic()#TODO:if error insert API key here

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
        output = self.dropout(roberta_output.last_hidden_state[:, 0, :])
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
    essays = []

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
            essays.extend(d["essay_text"])

    return np.mean(losses), np.vstack(preds), np.vstack(true_labels), essays

def get_gpt4_feedback(essay, scores):
    prompt = [
        {"role": "system", "content": "You are an AI assistant that provides feedback on essays based on given scores. Provide detailed feedback on the essay's grammar, vocabulary, coherence, and naturalness."},
        {"role": "user", "content": f"Essay: {essay}\n\nScores:\nGrammar: {scores[0]}/5\nVocabulary: {scores[1]}/5\nCoherence: {scores[2]}/5\nNaturalness: {scores[3]}/5\n\nPlease provide detailed feedback on this essay based on these scores."}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
    )
    return response.choices[0].message.content

def evaluate_feedback(essay, feedback):
    message = clientC.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are part of an educational research team analyzing the speaking skills of students in TOEFL ",

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are given an essay and feedback from a teacher for this essay. Your task is to evaluate the helpfulness of the feedback.

        Task: Evaluate the helpfulness of the feedback. Helpful feedback should explain what the errors are, why they are errors, and how to fix them.
        Give a score between 1 and 10, where 1 means the feedback is not helpful at all, and 10 means the feedback is very helpful.

        Essay: {essay}

        Feedback: {feedback}

        Please only provide a helpfulness score."""        
                    }
                ]
            }
    ]
    )
    return message.content[0].text

    # prompt = [
    #     {"role": "system", "content": "You are an AI assistant that evaluates the helpfulness of essay feedback."},
    #     {"role": "user", "content": f"""You are given an essay and feedback from a teacher for this essay. Your task is to evaluate the helpfulness of the feedback.

    #     Task: Evaluate the helpfulness of the feedback. Helpful feedback should explain what the errors are, why they are errors, and how to fix them.
    #     Give a score between 1 and 10, where 1 means the feedback is not helpful at all, and 10 means the feedback is very helpful.

    #     Essay: {essay}

    #     Feedback: {feedback}

    #     Please only provide a helpfulness score."""}
    # ]
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=prompt,
    # )
    # return response.choices[0].message.content

def extract_score(evaluation):
    try:
        score_line = evaluation.split('\n')[0]
        score = int(score_line.split(':')[1].strip())
        return score if 1 <= score <= 10 else None
    except:
        return None
    
def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EssayScoringModel(n_outputs=4, model_name=PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    loss_fn = torch.nn.MSELoss().to(device)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        train_loss, _, _ = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            len(train_df)
        )

        print(f'Train loss {train_loss}')

        val_loss, val_preds, val_labels, essays = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(test_df)
        )

        print(f'Val loss {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    model.load_state_dict(torch.load('best_model.pt'))
    _, val_preds, val_labels, essays = eval_model(model, test_data_loader, loss_fn, device, len(test_df))

    num_essays_to_evaluate = min(50, len(essays))
    selected_indices = random.sample(range(len(essays)), num_essays_to_evaluate)

    feedback_scores = []
    
    for i, index in enumerate(selected_indices):
        essay = essays[index]
        scores = val_preds[index]
        rounded_scores = [round(score) for score in scores]
        feedback = get_gpt4_feedback(essay, rounded_scores)
        feedback_evaluation = evaluate_feedback(essay, feedback)
        
        score = extract_score(feedback_evaluation)
        if score is not None:
            feedback_scores.append(score)
        
        print(f"\nProcessed essay {i+1}/{num_essays_to_evaluate}")

        # print(f"\nEssay {i+1}:")
        # print(f"Scores: Grammar: {rounded_scores[0]}, Vocabulary: {rounded_scores[1]}, Coherence: {rounded_scores[2]}, Naturalness: {rounded_scores[3]}")
        # print("GPT-4 Feedback:")
        # print(feedback)
        # print("\nFeedback Evaluation:")
        # print(feedback_evaluation)

    # Calculate frequency of feedback helpfulness scores
    score_frequency = Counter(feedback_scores)
    
    print("\nFeedback Helpfulness Score Frequencies:")
    for score in range(1, 11):
        frequency = score_frequency.get(score, 0)
        percentage = (frequency / len(feedback_scores)) * 100 if feedback_scores else 0
        print(f"Score {score}: {frequency} ({percentage:.2f}%)")


    average_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
    print(f"\nAverage Feedback Helpfulness Score: {average_score:.2f}")




def get_gpt4_feedback(essay):
    prompt = [
        {"role": "system", "content": "You are an AI assistant that provides feedback on essays. Provide detailed feedback on the essay's grammar, vocabulary, coherence, and naturalness."},
        {"role": "user", "content": f"Essay: {essay}\n\nPlease provide detailed feedback on this essay, focusing on grammar, vocabulary, coherence, and naturalness."}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
    )
    return response.choices[0].message.content

def evaluate_feedback(essay, feedback):
    message = clientC.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are part of an educational research team analyzing the speaking skills of students in TOEFL ",

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are given an essay and feedback from a teacher for this essay. Your task is to evaluate the helpfulness of the feedback.

        Task: Evaluate the helpfulness of the feedback. Helpful feedback should explain what the errors are, why they are errors, and how to fix them.
        Give a score between 1 and 10, where 1 means the feedback is not helpful at all, and 10 means the feedback is very helpful.

        Essay: {essay}

        Feedback: {feedback}

        Please only provide a helpfulness score."""        
                    }
                ]
            }
    ]
    )
    return message.content[0].text

def extract_score(evaluation):
    try:
        score_line = evaluation.split('\n')[0]
        score = int(score_line.split(':')[1].strip())
        return score if 1 <= score <= 10 else None
    except:
        return None

def generate_feedback():
    df = pd.read_csv('generated_essays_500.csv')
    essays = df['Essay'].tolist()

    num_essays_to_evaluate = min(50, len(essays))
    selected_indices = random.sample(range(len(essays)), num_essays_to_evaluate)
    selected_essays = [essays[i] for i in selected_indices]

    feedback_scores = []

    print("\n--- Generating Feedback for 50 Random Essays ---")
    for i, essay in enumerate(selected_essays):
        feedback = get_gpt4_feedback(essay)
        feedback_evaluation = evaluate_feedback(essay, feedback)
        
        score = extract_score(feedback_evaluation)
        if score is not None:
            feedback_scores.append(score)
        
        # print(f"\nEssay {i+1}:")
        # print("Essay content:")
        # print(essay[:500] + "..." if len(essay) > 500 else essay)  # Print first 500 characters of the essay
        # print("\nGPT-4 Feedback:")
        # print(feedback)
        # print("\nFeedback Evaluation:")
        # print(feedback_evaluation)
        # print("\n" + "-"*50)

    score_frequency = Counter(feedback_scores)  
    
    print("\nFeedback Helpfulness Score Frequencies:")
    for score in range(1, 11):
        frequency = score_frequency.get(score, 0)
        percentage = (frequency / len(feedback_scores)) * 100 if feedback_scores else 0
        print(f"Score {score}: {frequency} ({percentage:.2f}%)")

    average_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
    print(f"\nAverage Feedback Helpfulness Score: {average_score:.2f}")

if __name__ == '__main__':
    df = pd.read_csv('generated_essays_500.csv')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    PRE_TRAINED_MODEL_NAME = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    MAX_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 10

    train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

    train_and_evaluate()
    generate_feedback()