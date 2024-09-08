import whisper
import os
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def get_gpt4_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
    )
    return response.choices[0].message.content

def summarize_conversation(conversation_history):
    summary_prompt = [
        {"role": "system", "content": "Summarize the following conversation:"},
        {"role": "user", "content": "\n".join([msg['content'] for msg in conversation_history if msg['role'] == 'user'])}
    ]
    return get_gpt4_response(summary_prompt)

def text_to_speech(text, output_path):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    with open(output_path, 'wb') as f:
        f.write(response.content)

def process_mp3(mp3_path, conversation_history):
    wav_path = "temp_audio.wav"
    convert_mp3_to_wav(mp3_path, wav_path)
    transcription = transcribe_audio(wav_path)
    print("Transcription:", transcription)
    conversation_history.append({"role": "user", "content": transcription})
    # conversation_history.append({"role": "system", "content": "Please provide feedback on the English level of the user based on the transcription above. "})
    gpt4_response = get_gpt4_response(conversation_history)
    print("GPT-4 Response:", gpt4_response)
    conversation_history.append({"role": "assistant", "content": gpt4_response})
    output_path = "gpt_feedback.mp3"
    text_to_speech(gpt4_response, output_path)
    return conversation_history

initial_prompt = [
    # {"role": "system", "content": "You are an AI assistant that evaluates the English proficiency of users based on their speech. Provide detailed feedback on their language skills, including grammar and vocabulary. "}
    {"role": "system", "content": "You are an expert in evaluating English language proficiency based on TOEFL speaking tasks. Your task is to score the following transcription of a spoken conversation according to TOEFL standards. Consider the following criteria for evaluation: grammar, vocabulary, coherence, and naturalness of the conversation. Provide a score out of 5 for each criterion, being particularly stringent, and give detailed suggestions on areas for improvement."}
]

test_prompt = [
    {
  "role": "system",
  "content": "Act like an expert in evaluating English language proficiency based on TOEFL speaking tasks. You have been evaluating TOEFL speaking responses for over 15 years and are well-versed in the TOEFL scoring criteria. Your task is to score the following transcription of a spoken conversation according to TOEFL standards. The evaluation should be based on four key criteria: grammar, vocabulary, coherence, and naturalness of the conversation. For each criterion, provide a score out of 5, adhering to a stringent evaluation approach to ensure the highest accuracy in scoring."
},
{
  "role": "system",
  "content": "Objective:\n1. Grammar: Evaluate the accuracy and appropriateness of grammatical structures used. Consider aspects such as subject-verb agreement, tense consistency, and the correct use of complex sentences.\n2. Vocabulary: Assess the range and appropriateness of vocabulary used. Consider the precision of word choice, the use of idiomatic expressions, and the overall richness of the language.\n3. Coherence: Examine the logical flow and clarity of the conversation. Assess how well the response is organized, the presence of clear main ideas, and the effectiveness of transitions between ideas.\n4. Naturalness: Evaluate the naturalness and fluency of the conversation. Consider the speaker's ease of expression, the use of intonation and stress, and how naturally the language flows."
},
{
  "role": "system",
  "content": "Steps:\n1. Read the transcription carefully: Ensure you understand the content and context of the conversation.\n2. Score each criterion separately: For each of the four criteria (grammar, vocabulary, coherence, and naturalness), provide a score out of 5. Be particularly stringent in your scoring to ensure a detailed and precise evaluation.\n3. Provide detailed suggestions for improvement: For each criterion, offer specific and actionable feedback. Highlight areas where the speaker performed well and identify specific aspects that need improvement. Offer examples or alternatives where possible to illustrate your points.\n4. Summarize the overall performance: Conclude with a summary that encapsulates the strengths and weaknesses of the response, based on your detailed evaluations."
},
{
  "role": "system",
  "content": "Example Format:\n- Grammar: 3.5/5\n  - Suggestions: Improve the use of complex sentences and ensure tense consistency.\n- Vocabulary: 4/5\n  - Suggestions: Incorporate more idiomatic expressions and precise word choices.\n- Coherence: 3/5\n  - Suggestions: Enhance the logical flow by using clearer transitions between ideas.\n- Naturalness: 3.5/5\n  - Suggestions: Work on the natural flow of language and appropriate intonation."
},
{
  "role": "system",
  "content": "Take a deep breath and work on this problem step-by-step."
}
]

# conversation_history = initial_prompt
conversation_history = test_prompt

while True:
    mp3_path = input("Enter the path to the MP3 file (or 'q' to exit): ")
    if mp3_path.lower() == 'q':
        break
    
    conversation_history = process_mp3(mp3_path, conversation_history)
    
    # if len(conversation_history) > 2000:
    #     summarized_history = summarize_conversation(conversation_history)
    #     conversation_history = [{"role": "system", "content": summarized_history}]
    #     print("Conversation summarized to keep it concise.")
