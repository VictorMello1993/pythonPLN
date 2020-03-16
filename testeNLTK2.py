import nltk
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


def RemoveStopWords(texto):
    stopwords = set(nltk.corpus.stopwords.words('portuguese')) #Remoção de stopwords em português
    palavras = [i for i in texto.split() if not i in stopwords]
    print('Resultado da função RemoveStopWords: '," ".join(palavras))    


def Stemming(texto):
    stemmer = nltk.stem.RSLPStemmer() #Stemizador de palavras na língua portuguesa
    palavras = []
    for palavra in texto.split():
        palavras.append(stemmer.stem(palavra))
    print('Resultado da função Stemming: '," ".join(palavras))    


# Removendo links usando expressões regulares
def LimpezaDados(texto):
    texto = re.sub(r"http\S+", "", texto).lower().replace('.', '').replace(';', '').replace('-', '').replace('(', '').replace(')', '').replace('|', '').replace('_', '').replace(':', '')
    print('Resultado da função LimpezaDados: ', texto)

RemoveStopWords('Eu não gosto da Apple, mas tudo o que eu gosto são de celulares do Android')
Stemming('Olá, meu nome é Victor Mello')
LimpezaDados('Assista aqui o vídeo do Governador falando sobre Coronavírus https://g1.globo.com/rj/rio-de-janeiro/noticia/2020/03/13/witzel-cria-gabinete-de-crise-para-combater-o-novo-coronavirus-no-rj.ghtml :) ;)')
