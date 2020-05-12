#Text to Speech utilizando Google Text to Speech

import speech_recognition as sr
from gtts import gTTS #Google Text to Speech
import os
import subprocess

message = 'Olá, Victor! Como eu posso te ajudar?'
tts = gTTS(message, lang='pt')
tts.save('teste.mp3')
subprocess.call(['notepad.exe', r'D:\Documentos\Monografia I\Monografia\Projetos\pythonpln\teste.mp3'])
# os.system('teste.mp3')