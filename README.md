### Let's Speak French :)

Learning a new language is hard. Practicing speaking a new language is harder, especially if you don't have any willing francophone friends.

This project helps you practice speaking, with a bot, using open-source LLMs. You can speak, get an LLM response (in audio and then text), then continue a conversation. 

There are also vocabulary assists, and the prompt can be changed for different types of conversions.

There's no substitute for a qualified teacher and/or patient native speakers, but getting some live audio responses with questions can be quite helpful for practicing dialogues / getting more comfortable expressing ideas in your target language.

### Setup

1. Clone this repo, create a virtual environment with Python 3.11.1
2. pip install the requirements.txt file
3. You may need to allow your python editor to access your laptop microphone - depending on your device
4. Input a valid HuggingFace Access Token in main.py, so you can use the Huggingface APIs
5. You should be good to go - just run main.py to start. The first time you run this, it will take some time as the whisper-small audio recognition model will get downloaded locally.
   
