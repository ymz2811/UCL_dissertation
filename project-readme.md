# Automated Spoken Language Proficiency Assessment System

This project develops an automated system for assessing spoken language proficiency, focusing on TOEFL exam preparation. It combines speech recognition, natural language processing, and machine learning to transcribe speech, evaluate language proficiency, and provide personalized feedback.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Results](#results)

## Overview

This system aims to address challenges in traditional language assessment methods by providing a scalable, consistent, and accessible solution for evaluating spoken English proficiency. It includes components for speech recognition, language proficiency scoring, and feedback generation.

## Setup

1. Clone this repository:
   ```
   git clone [repository-url]
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   You need to set environment variables for the OpenAI and Anthropic API keys. Use the following commands based on your operating system:

   For Windows (Command Prompt):
   ```
   set OPENAI_API_KEY=your_openai_api_key_here
   set ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

   For Unix-based systems (Linux, macOS) or Windows PowerShell:
   ```
   export OPENAI_API_KEY=your_openai_api_key_here
   export ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

   Alternatively, you can create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## File Descriptions

- `generate_script.py`: Generates synthetic data for training and testing. Note that this script will generate a sample file of 10 essays.
- `BERT_base.py`, `RoBERTa_large.py`, `RoBERTa_plot.py`: Scripts for testing model proficiency using different architectures.
- `process_audio.py`: Runs the complete model pipeline using LLM for scoring.
- `generated_essays_500.csv`: Contains 500 essays with synthetic scores.
- `speaking_topics.csv`: TOEFL questions used for synthetic data generation.

## Usage

1. Generate synthetic data (sample of 10 essays):
   ```
   python generate_script.py
   ```
   Note: This will create a file named `generated_essays_10.csv` with 10 sample essays.

2. Run model proficiency tests:
   ```
   python BERT_base.py
   python RoBERTa_large.py
   python RoBERTa_plot.py
   ```
   These scripts use the full dataset (`generated_essays_500.csv`) for testing.

3. Process audio and generate feedback:
   ```
   python process_audio.py
   ```
   Note: You will need an MP3 file containing the spoken response to process. Ensure that you have an MP3 file ready before running this script.

## Results

Our experiments show that:
- The Whisper model performs well in speech recognition across various languages.
- Fine-tuned models (BERT, RoBERTa) outperform general-purpose LLMs in language proficiency evaluation.
- Increasing dataset size beyond 100 essays doesn't significantly improve model performance.

For detailed results and analysis, please refer to the accompanying research paper.

