# Transcrevendo um arquivo de áudio para texto

import speech_recognition as sr

r = sr.Recognizer()

arquivo_voz = 'voz_1.wav'

with sr.AudioFile(arquivo_voz) as fonte:
    audio = r.record(fonte)
    
    print(r.recognize_google(audio, language='pt-BR'))