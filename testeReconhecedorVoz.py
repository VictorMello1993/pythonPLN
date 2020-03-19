import speech_recognition as sr

r = sr.Recognizer()

# Utilizando microfone como fonte de áudio
with sr.Microphone() as fonte:
    r.adjust_for_ambient_noise(fonte)  # Cancelando ruídos
    print('Diga alguma coisa: ')
    while True:
        audio = r.listen(fonte)  # Captura de voz
        texto = r.recognize_google(audio, language='pt-BR')
        try:
            print('Você disse: ' + texto)
        except sr.UnknownValueError:
            print('Não foi possível compreender o que você disse')
        except sr.RequestError as e:
            print('Erro ao executar recognize_google, {0}'.format(e))
