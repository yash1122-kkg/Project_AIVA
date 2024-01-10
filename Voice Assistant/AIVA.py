import speech_recognition as sr
import os
import pyttsx3
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')

import openai
openai.api_key = OPENAI_KEY

def SpeechText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

r = sr.Recognizer()

def record_text():
    while(1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("I'm Listening")
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                return MyText
        except sr.RequestError as e:
            print("Could not Request result: {0}".format(e))
        except sr.UnknownValueError:
            print("Unknown vlaue error")

def send_to_GPT(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        max_tokens = 100,
        n = 1,
        stop = None,
        temperature = 0.5,
    )

    messages = response.choice[0].message.content
    messages.append(response.choice[0].messages)
    return messages

messages = [{"role": "user", "content": "Please act like Jarvis from Iron man."}]
while(1):
    text = record_text()
    messages.append({"role": "user", "content": text})
    response = send_to_GPT(messages)
    SpeechText(response)

    print(response)