import speech_recognition as sr

r = sr.Recognizer()

#Utilizando microfone para capturar voz
with sr.Microphone() as fonte:
    print('Diga alguma coisa: ')
    while True:
        audio = r.listen(fonte) #Captura de voz
        texto = r.recognize_google(audio, 'pt-BR') 

        with open('voz.wav', 'wb') as arquivo:
            arquivo.write(audio.get_wav_data()) #Salvando a voz reconhecida em um arquivo wav
        try:
            print('Voce disse: ' + texto)
        except:
            print('Não foi possível compreender o que você disse')