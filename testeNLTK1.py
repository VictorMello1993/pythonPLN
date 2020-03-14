import nltk

texto = 'Hello, My name is Victor. How are you?'

#Dividindo uma string em array de substrings quando tiver um ponto . no meio da frase
frases = nltk.tokenize.sent_tokenize(texto)
print(frases)


#Tokenizando uma frase
tokens = nltk.word_tokenize(texto)
print(tokens)

'''Obtendo uma lista de tuplas onde cada elememto 
corresponde a um par de palavras e suas respectivas classes 
gramaticais'''

classes = nltk.pos_tag(tokens)
print(classes)

#Para mais informações sobre as classes gramaticais, visitar o site https://cs.nyu.edu/grishman/jet/guide/PennPOS.html

#Detecção de entidades
entidades = nltk.chunk.ne_chunk(classes)
print(entidades)

'''OBS: NLTK não suporta bem em português. Mesmo colocando o parâmetro language='portuguese' na função
   sent_tokenize(), o algoritmo trata a tokenização como se as palavras estivessem em inglês'''
