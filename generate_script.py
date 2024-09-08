import os
import pandas as pd
import random
from openai import OpenAI
# from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')


client = OpenAI(api_key=api_key)

df = pd.read_csv('speaking_topics.csv', nrows=10, on_bad_lines='skip')

def generate_scores():
    grammar = random.choices([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.3, 0.1])[0]
    vocabulary = grammar + random.choice([-1, 0, 1])
    coherence = grammar + random.choice([-1, 0, 1])
    naturalness = grammar + random.choice([-1, 0, 1])
    
    vocabulary = max(1, min(vocabulary, 5))
    coherence = max(1, min(coherence, 5))
    naturalness = max(1, min(naturalness, 5))
    
    return grammar, vocabulary, coherence, naturalness

questions = df['Topics;'].tolist()
scores = [generate_scores() for _ in questions]

def create_prompt(question, grammar_score, vocabulary_score, coherence_score, naturalness_score):
    prompt = f"""
    You will need to generate a simulated speaking response to the question: "{question}" 

    The response should reflect the following rubric scores:
    - Grammar: {grammar_score}
    - Vocabulary: {vocabulary_score}
    - Coherence: {coherence_score}
    - Naturalness of the conversation: {naturalness_score}

    Write an essay that demonstrates a student's spoken answer based on these scores. Remember to only give out the simmulated response. Remember to make intentional mistakes when the score is not perfect.

    Take a deep breath and work on this problem step-by-step, making sure to make clear mistakes where the score is lowered.
    """
    return prompt

essays = []

for question, (grammar, vocabulary, coherence, naturalness) in zip(questions, scores):
    prompt = create_prompt(question, grammar, vocabulary, coherence, naturalness)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides spoken answers to questions."},
            {"role": "user", "content": prompt},
        ],
    )
    
    essay = response.choices[0].message.content
    essays.append({
        "Question": question, 
        "Essay": essay,
        "Grammar Score": grammar,
        "Vocabulary Score": vocabulary,
        "Coherence Score": coherence,
        "Naturalness Score": naturalness
    })

essays_df = pd.DataFrame(essays)
essays_df.to_csv("generated_essays_10.csv", index=False)