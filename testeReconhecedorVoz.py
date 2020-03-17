
import speech_recognition as sr

r = sr.Recognizer()

#Utilizando microfone para capturar voz
with sr.Microphone as fonte:
    print('Diga alguma coisa: ')
    audio = r.listen(fonte) #Captura de voz

    texto = r.recognize_google(audio)

    try:
        print('Você disse: ' + texto)
    except:
        print('Não foi possível compreender o que você disse')