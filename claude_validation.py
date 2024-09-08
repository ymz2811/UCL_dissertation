import numpy as np
import anthropic
import warnings
import pandas as pd
import re
from anthropic import Anthropic
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score


df = pd.read_csv('generated_essays_500.csv')


client = Anthropic()#TODO: If error, insert API key here


def analyze_essay(essay):
    message = client.messages.create(
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
                        "text": f"""Analyze, for a level of TOEFL, the following speaking essay for grammar, vocabulary, coherence, and naturalness of speech. Provide brief feedback for each category, followed by a predicted score from 1 to 5 (1 being poor, 5 being excellent).
                        
                        Essay: {essay}

                        Format your response as follows:
                        Grammar:
                        [Feedback]
                        Predicted score: [1-5]

                        Vocabulary:
                        [Feedback]
                        Predicted score: [1-5]

                        Coherence:
                        [Feedback]
                        Predicted score: [1-5]

                        Naturalness of speech:
                        [Feedback]
                        Predicted score: [1-5]
                        """             
                    }
                ]
            }
        ]
    )
    return message.content[0].text




results = []

for index, row in df.iterrows():
    essay = row['Essay']
    analysis = analyze_essay(essay)
    
    results.append({
        'essay_id': index,
        'original_essay': essay,
        'claude_analysis': analysis
    })
    if index % 10 == 0:
        print(f"Processed essay {index + 1}/{len(df)}")

results_df = pd.DataFrame(results)

final_df = pd.concat([df, results_df], axis=1)

final_df.to_csv('analyzed_essays.csv', index=False)

def extract_scores(analysis):
    scores = {}
    categories = ['Grammar', 'Vocabulary', 'Coherence', 'Naturalness of speech']
    
    for category in categories:
        match = re.search(f"{category}:.*?Predicted score: (\d)", analysis, re.DOTALL)
        if match:
            scores[category.lower()] = int(match.group(1))
    
    return scores

final_df['claude_scores'] = final_df['claude_analysis'].apply(extract_scores)

final_df = pd.concat([final_df, final_df['claude_scores'].apply(pd.Series)], axis=1)


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
final_df = final_df.rename(columns={"naturalness of speech": "naturalness"})
final_df.to_csv('data.csv', index=False)
for category in ['Grammar', 'Vocabulary', 'Coherence', 'Naturalness']:
    original_scores = final_df[f'{category} Score']
    claude_scores = final_df[category.lower()]
    
    correlation, _ = pearsonr(original_scores, claude_scores)
    print(f"{category} correlation: {correlation:.2f}")
    
    mae = np.mean(np.abs(original_scores - claude_scores))
    print(f"{category} Mean Absolute Error: {mae:.2f}")
    
    weights = 'quadratic' 
    kappa = cohen_kappa_score(original_scores, claude_scores, weights=weights)
    print(f"{category} Weighted Kappa Score: {kappa:.2f}")
    
    print()