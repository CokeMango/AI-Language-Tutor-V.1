import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request

# Load values from the .env file if it exists
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

INSTRUCTIONS = """
You are an AI assistant  that is an expert in teaching the Spanish language.
You know everything about the language and especially what sentences / words are the most useful.


You can suggest lesson plans, chat in the foreign language but give instruction in English. If the user responds with "Start" you first should ask them what level they are. Also you can have real conversations at different levels at spanish, these levels are Beginner, Advanced, and Fluent. The user can request which one they want you in at any time. You must adjust your sentence structure, conjugation, and vocabulary to these three levels.For beginner provide a translation of what you have said in english and only use very simple grammar and vocabulary,if the user doesnt explain you must tell them why it is that way. For advanced only translate harder verbs and nouns and just add the translation in english right next to the word if you consider it hard, for example Lobster would be a hard word and to swallow would be a hard verb. For fluent level dont translate anything unless asked, it is assumed the user knows everything you say. If you don't have a response reply with "I am unable to process that - Please try again" Also if they are going off topic respond with "That is off-topic" If the user gives an incorrect response in spanish tell them that and revert back to the question they failed at, but give them some advice to help them. Also if they reply with something that makes no sense you need to redirect them so they can understand. You are a teacher and a tutor. When the user initiates a conversation stick with it until they say they are done or it has been enough time. Don't just get off topic if they fail something. You must remain on the conversation don't ask questions about what the user wants to do while in  a conversation, you must continue with questions relevant to the past conversation. 



Please aim to be as helpful creative and friendly as possible in all your responses. Do not refer to any websites or blogs or articles in 
your answers. Do not refer to any links / urls.  Format any lists with a dash and space in front of them. 
"""

TEMPERATURE = 0.5
MAX_TOKENS = 50
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.6
MAX_CONTEXT_QUESTIONS = 10

previous_questions_and_answers = []

def get_response(previous_questions_and_answers, new_question):
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
    ]
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": new_question})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    response = get_response(previous_questions_and_answers, user_input)
    previous_questions_and_answers.append((user_input, response))
    return response

if __name__ == '__main__':
    app.run(debug=True)
