# %% [markdown]

# Análise de sentimentos usando Machine Learning

# Criando modelos para análise de sentimentos de tweets

from nltk import word_tokenize

import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# %% [markdown]
# Importando API do Twitter
from TwitterSearch import *

try:
    ts = TwitterSearch(
        consumer_key='xxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        consumer_secret='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        access_token='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
        access_token_secret='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    tso = TwitterSearchOrder()
    tso.set_keywords(['iphone', 'android'])
    tso.set_language('pt')

    for tweet in ts.search_tweets_iterable(tso):
        print('@%s tweeted: %s' %
              (tweet['user']['screen_name'], tweet['text']))

except TwitterSearchException as e:
    print(e)
# %%
# Obtendo uma base de dados
df = pd.read_csv('Tweets_Mg.csv', encoding='utf-8')
df.head()
# %% [markdown]
# Conta a quantidade de linhas de tweets neutros, positivos e negativos
df.Classificacao.value_counts()
# %% [markdown]
# Mostrando o número de tweets positivos, negativos e neutros graficamente
%matplotlib inline
df.Classificacao.value_counts().plot(kind='bar')
# %%
# Mostrando o total de linhas de uma base de dados
df.count()

# %% [markdown]
# Pré-processamento de dados
# Remove linhas duplicadas
# Remove stopwords
# Técnica de stemming ou lemmatization
# Remove caracteres indesejados como links

# Removendo tweets duplicados
df.drop_duplicates(['Text'], inplace=True)
# %%
df.Text.count() #O número total de tweets será reduzido após a remoção de duplicatas
# %% [markdown]
# Separando tweets e suas classes
tweets = df['Text']
classes = df['Classificacao']
# %%
# %% [markdown]
# Instala bibliotecas e baixa a base de dados
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
nltk.download('wordnet')
# %% [markdown]
# Funções de pré-processamento de dados



def RemoveStopWords(texto):
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in texto.split() if not i in stopwords]
    return " ".join(palavras)


def Stemming(texto):
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for palavra in texto.split():
        palavras.append(stemmer.stem(palavra))
    return " ".join(palavras)

# Removendo links através de um regex


def LimpezaDados(texto):
    texto = re.sub(r"http\S+", "", texto).lower().replace('.', '').replace(';', '').replace(
        '-', '').replace('(', '').replace(')', '').replace('|', '').replace('_', '').replace(':', '')
    return texto


# %% [markdown]
from nltk.stem import WordNetLemmatizer
worldnet_lemmatizer = WordNetLemmatizer()


def lemmatization(texto):
    palavras = []
    for palavra in texto.split():
        palavras.append(worldnet_lemmatizer.lemmatize(palavra))
    return "".join(palavras)
# %% [markdown]

# # Teste funcionamento das funções de pré-processamento

# Removendo stopwords
RemoveStopWords('Eu não gosto do partido, e também não votaria novamente nesse governante!')
Stemming('Eu não gosto do partido, e também não votaria novamente nesse governante!')
LimpezaDados('Assista aqui o vídeo do Governador falando sobre CEMIG https://www.uol.com.br :) ;)')

# %%
# %% [markdown]
## Aplicando as funções de pré-processamento em uma única função
def Preprocessing(texto):
    stemmer = nltk.stem.RSLPStemmer()
    texto = re.sub(r"http\S+", "", texto).lower().replace('.', '').replace(';', '').replace('-', '').replace('(', '').replace(')', '').replace('|', '').replace('_', '').replace(':', '')
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [stemmer.stem(i) for i in texto.split() if not i in stopwords]
    return " ".join(palavras)

# %%
tweets = [Preprocessing(i) for i in tweets]

# OBS: o tokenizador padrão da NLTK é bastante limitado a certos
# idiomas, como português, o que influencia muito no resultado
# final do processo de análise de sentimentos"""
# %%
tweets[:50]
# %% [markdown]
#Utilizando o tokenizador para trabalhar especificamente com tweets
from nltk.tokenize import TweetTokenizer
# %%
#Instanciando um tokenizador de tweets
tweet_tokenizer = TweetTokenizer()
# %%
teste = 'A live do @fulano é show ;-) :)'
tweet_tokenizer.tokenize(teste)
# %% [markdown]
# #Criando um modelo de treinamento do classificador de tweets

#Criando um vetorizador de tweets
vetorizador = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize)
# vetorizador = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize, max_features=1000) #Para bases de dados muito grandes
# %% [markdown]
# Aplica o vetorizador nos dados de texto
freq_tweets = vetorizador.fit_transform(tweets)
type(freq_tweets) #Resultado: matriz esparsa de tweets
# %%
freq_tweets.shape
# %% [markdown]

## Treinamento do algoritmo
modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)

# %%
freq_tweets.A

# %%
testes = ['Esse governo está no início, vamos ver no que vai dar',
          'Estou muito feliz com o governo de Minas esse ano',
          'O estado de Minas Gerais decretou calamidade financeira!!!',
          'A segurança desse país está deixando muito a desejar',
          'O governo de Minas mais uma vez é do PT']


# %% [markdown]
# Aplicando a função de pré-processamento nos dados de teste
testes = [Preprocessing(i) for i in testes]

# %%
# Transforma os dados de teste em vetores de palavras
freq_testes = vetorizador.transform(testes)

# %%
for t, c in zip(testes, modelo.predict(freq_testes)):
    print(t +", "+ c)
# %% [markdown]
# Exibindo as probabilidades de cada classe
print(modelo.classes_)
modelo.predict_proba(freq_testes).round(2)

# %%
## Tags de negações
# Acrescenta a tag _neg após encontrar um 'não' no meio da frase 
# Objetivo: dar mais peso para o modelo identificar uma inversão de sentimento da frase
#Exemplos:
    #Eu gosto de cachorros, positivo
    #Eu NÃO gosto de cachorros, negativo

def marque_negacao(texto):
    negacoes = ['não', 'not', 'NÃO', 'n', 'N', 'NOT']
    negacao_detectada = False
    resultado = []
    palavras = texto.split()
    for p in palavras:
        p = p.lower()
        if negacao_detectada == True:
            p = p + '_NEG'
        if p in negacoes:
            negacao_detectada = True
        resultado.append(p)
    return (" ".join(resultado))

# %%
marque_negacao('Eu gosto de heavy metal')
marque_negacao('Eu não gosto de heavy metal')

# %%
## Criando modelos de pipeline
#São utilizados para reduzir o código e automatizar fluxos

from sklearn.pipeline import Pipeline

# %%
pipeline_simples = Pipeline([
    ('counts', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# %% [markdown]
# Pipiline que atribui tags de negação nas palavras
pipeline_negacoes = Pipeline([
    ('counts', CountVectorizer(tokenizer=lambda text: marque_negacao(text))),
    ('classifier', MultinomialNB())
])


# %%
pipeline_simples.fit(tweets, classes)

# %%
#Exibindo as estapas do pipeline(1 - vetorização 2 - classificador)
pipeline_simples.steps

# %%
#Gerando um modelo de negações
pipeline_negacoes.fit(tweets, classes)

# %%
# Etapas do pipeline de negações
pipeline_negacoes.steps


# %% [markdown]
## Modelo com SVM
pipeline_svm_simples = Pipeline([
    ('counts', CountVectorizer()),
    ('classifier', svm.SVC(kernel='linear'))
])

pipeline_svm_negacoes = Pipeline([
    ('counts', CountVectorizer(tokenizer=lambda text: marque_negacao(text))),
    ('classifier', svm.SVC(kernel='linear'))
])

# %% [markdown]
## Validando os modelos com validação cruzada

# Fazendo o cross validation do modelo

resultados = cross_val_predict(pipeline_simples, tweets, classes, cv=10)
# %%[markdown]
#Medindo acurácia do modelo
metrics.accuracy_score(classes, resultados)
# %% [markdown]
# Medidas de avaliação do modelo
sentimento = ['Positivo', 'Negativo', 'Neutro']
print(metrics.classification_report(classes, resultados, sentimento))


# %% [markdown]
# Matriz de confusão
print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))

# %%
def Metricas(modelo, tweets, classes):
    resultados = cross_val_predict(modelo, tweets, classes, cv=10)
    return 'Acurácia do modelo: {}'.format(metrics.accuracy_score(classes, resultados))

# %% [markdown]
# Naive Bayes simples
Metricas(pipeline_simples, tweets, classes)


# %% [markdown]
# Naive Bayes com tag de negações
Metricas(pipeline_negacoes, tweets, classes)


# %% [markdown]
# SVM linear simples
Metricas(pipeline_svm_simples, tweets, classes)
# %%[markdown]
# SVM linear com tag de negações
Metricas(pipeline_svm_negacoes, tweets, classes)


# %% [markdown]
## Modelo com tag de negações
resultados = cross_val_predict(pipeline_negacoes, tweets, classes, cv=10) 
# %%
# Medindo acurácia média do modelo
metrics.accuracy_score(classes, resultados)
# %%
sentimento = ['Positivo', 'Negativo', 'Neutro']
print(metrics.classification_report(classes, resultados, sentimento))
# %%[markdown]
# Matriz de confusão
print(pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True))

# %% [markdown]
## Avaliando o modelo com Bigrams

# Exemplo: 'Eu gosto do Brasil' ----> 'eu gosto', 'gosto do', 'do Brasil'

#Bigrams
vetorizador = CountVectorizer(ngram_range=(2,2))
freq_tweets = vetorizador.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)

# %%
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
# %%
metrics.accuracy_score(classes, resultados)

# %%
