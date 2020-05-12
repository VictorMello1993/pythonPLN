#Reconhecedor de voz completo utilizando Google Text to Speech
#Fonte: https://pythonspot.com/personal-assistant-jarvis-in-python/

import speech_recognition as sr
from time import ctime
import time
import os
from gtts import gTTS
import subprocess

def speak(audioString):
    print(audioString)
    tts = gTTS(text=audioString, lang='pt-br')
    tts.save("audio.mp3")
    os.system("audio.mp3")

def recordAudio():
    r = sr.Recognizer()
    with sr.Microphone() as source: #Utilizando microfone como fonte de áudio
        print("Diga alguma coisa:")
        audio = r.listen(source)
    
    data = ""
    try:
        # Uses the default API key
        # To use another API key: `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        data = r.recognize_google(audio, language='pt-BR')
        print("Você disse: " + data)
    except sr.UnknownValueError:
        print("Não foi possível compreender o que você disse")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return data

def jarvis(data):
    if "como vai você" in data:
        speak("Estou bem, e você?")

    if "Que horas são" in data:
        speak(ctime())

    if "Onde fica" in data:
        data = data.split(" ")
        location = data[2]
        speak("Aguarde um pouco Victor, Irei te mostrar onde " + location + " se localiza.")        
        subprocess.call(['Google Chrome.exe', "https://www.google.nl/maps/place/" + location + "/&amp;"])        
        # os.system("chromium-browser https://www.google.nl/maps/place/" + location + "/&amp;")

# Início do programa
time.sleep(2)
speak("Olá, Victor! Como eu posso te ajudar?")
while 1:
    data = recordAudio()
    jarvis(data)