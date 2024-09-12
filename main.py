import os

import sounddevice as sd
from openai import OpenAI
import time
import pyttsx3
from transformers import pipeline
import wave
import numpy as np
import warnings

import threading





from transformers import pipeline, GenerationConfig

pipe = pipeline("automatic-speech-recognition", model="sanchit-gandhi/whisper-small-fr-1k-steps",
                  device =0,
                  )

generation_config = GenerationConfig.from_pretrained("openai/whisper-medium")
pipe.model.generation_config = generation_config



API_KEY = 
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"


engine = pyttsx3.init()
rate = engine.getProperty('rate')
#engine.setProperty('rate', rate-25)
engine.setProperty('voice', 'com.apple.voice.compact.fr-FR.Thomas')

global loop_running
loop_running = True

def run_loop():
    while loop_running:

        time.sleep(2)

# Function to stop the loop when user input is given
def stop_loop():
    global loop_running
    while loop_running:
        k = input('')
        if '1' in k:
            loop_running = False


def record_audio(filename, fs):
    print("Recording... Input '1' to stop.")
    audio_data = []
    stop_recording = [False]

    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    # Start recording
    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback)
    global loop_running
    loop_running = True
    with stream:
        loop_thread = threading.Thread(target=run_loop)
        
        loop_thread.start()

        # Wait for user input to stop the loop
        stop_loop()

        # Ensure the loop thread stops
        loop_thread.join()

        #print("Loop has stopped.")

    audio_data = np.concatenate(audio_data, axis=0)
    print("Recording finished")

    # Save as WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved as {filename}")

def llm_output(transcription, history):
    #watch a video 
    prompt = """Vous êtes un professeur de français serviable et avez une conversation pratique avec un étudiant français. 
    Répondez avec un message en français. 
    soyez conversationnel et décontracté, et corrigez les erreurs grammaticales si elles se produisent. 
    Commencez toujours par une question qui peut aider l'élève à parler et debatter."""

    #discuss something from your life as an interview
    prompt = """
    Vous êtes un professeur de français serviable et avez une conversation pratique avec un étudiant français. 
    Répondez avec un message en français. 
    Vous jouez un exercice de conversation, dans le jeu de rôle vous êtes un interviewer qui m'interroge sur ma vie. Posez des questions qui peuvent susciter des réactions émotionnelles et des opinions, soyez curieux.
    Commencez toujours par une question qui peut aider moi à parler et debatter.
    """

    me = transcription

    # init the client but point it to TGI
    client = OpenAI(
        # replace with your endpoint url, make sure to include "v1/" at the end
        base_url=f"{API_URL}/v1/",
        # replace with your API key
        api_key=API_KEY,
    )

    opener = "notre conversation jusqu'à présent: \n"

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": opener + history + "\nProchaine dialogue:\n" + me}
        ],
        stream=False,
        max_tokens=1000,
        temperature=0.5
    )

    out = chat_completion.choices[0].message.content
    

    engine.say(out)
    engine.runAndWait()
    print(out)
    

    history += f"\n moi: {transcription} \n vous: {out}"

    return out, history

def vocabulary_output(history):
    prompt = """
    you are a french professor. this is the conversation history between your student and yourself.
    create a short list of vocabulary / phrases that are relevent to the conversation history, in both french and english.
    do not repeat words that are in the history, find new words and phrases that can be relevant to the topic"""
    client = OpenAI(
        # replace with your endpoint url, make sure to include "v1/" at the end
        base_url=f"{API_URL}/v1/",
        # replace with your API key
        api_key=API_KEY,
    )

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history}
        ],
        stream=False,
        max_tokens=1000,
        temperature=1
    )

    out = ''
    print(chat_completion.choices[0].message.content)
    print(out)

def on_press_pause(key):
    global pause
    try:
        if key.char == 'p':
            pause = False
            return False  # Stop listener
    except AttributeError:
        pass


if __name__ == "__main__":
    
    history = ''
    fs = 44100  # Sample rate
    #print(pipe.model.generation_config)
    cont = True
    i = 0
    while cont:
        filename = f"session_audio/output{i}.wav"
        record_audio(filename, fs)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transcript = pipe(filename)['text']
            print(f'transcription\n{transcript}')
        print('Time taken to transcribe:',time.time() - start)

        out, history = llm_output(transcript, history)

        i+=1

        k = input('\n\n 1 to stop, 2 for vocab, any button to continue \n \n')
        if '1' in k:
            cont = False
            #return 0
            vocabulary_output(history)
            exit()
        elif '2' in k:
            vocabulary_output(history)
            time.sleep(30)
            k = ''

            
