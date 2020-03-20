import speech_recognition as sr

r = sr.Recognizer()

# Utilizando microfone como fonte de áudio
with sr.Microphone() as fonte:
    copia_arquivo = 1
    r.adjust_for_ambient_noise(fonte)  # Cancelando ruídos
    print('Diga alguma coisa: ')
    while True:
        audio = r.listen(fonte)  # Captura de voz

        # Salvando a voz reconhecida em um arquivo wav
        with open(f'voz_{copia_arquivo}.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        texto = r.recognize_google(audio, language='pt-BR') #Reconhecendo voz em português
        copia_arquivo += 1
        try:
            print('Você disse: ' + texto)
        except sr.UnknownValueError:
            print('Não foi possível compreender o que você disse')
        except sr.RequestError as e:
            print('Erro ao executar recognize_google, {0}'.format(e))
