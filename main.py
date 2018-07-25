import os
import preprocessing
import polarity
import features
import ngramGenerator
import numpy as np
from pprint import pprint
from sklearn import preprocessing as pr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def abs_file_url(filename):
    current_directory = os.path.dirname(__file__)
    return os.path.join(current_directory, filename)

def mapTweet(tweet, afinn, emoDict, positive, negative, neutral, slangs):
    out=[]
    line = preprocessing.processTweet(tweet, stopWords, slangs) # limpio el tweet, eliminando las palabras innecesarias y sobreescribiendo los modismos
    out.append(polarity.afinnPolarity(line, afinn)) # afinidad
    out.append(float(features.emoticonScore(line, emoDict))) # emoticon score
    out.append(float(features.hashtagWordsRatio(line))) # porcentaje de palabras con hashtag
    out.append(float(len(line)/140)) # tamaño total de los 140 carácteres utilizados
    out.append(float(features.upperCase(line))) # si existen mayúsuculas en el tweet; 1 = si, 0 = no
    out.append(float(features.exclamationTest(line))) # si tiene signo de exclamación o no; 1 = si, 0 = no
    out.append(float(line.count("!")/140)) # procentaje de signos de exlamación
    out.append(float((features.questionTest(line)))) # si tiene un signo de pregunta
    out.append(float(line.count('?')/140)) # procentaje de signos de preguntas
    out.append(float(features.freqCapital(line))) # porcentaje de las letras en mayusculas
    u = features.scoreUnigram(line, positive, negative, neutral) # Score sobre el vector de palabras utilizadas en los documentos de prueba
    out.extend(u)
    return out

def loadData(vectors, labels, file_url, label):
    f = open(file_url,'r', encoding="utf-8", errors="ignore")
    line = f.readline()
    while line:
        tweet_mapped = mapTweet(line, afinn, emoticonDict, positive, negative, neutral, slangs)
        vectors.append(tweet_mapped)
        labels.append(float(label))
        line = f.readline()
    f.close()

def predecir(tweet, model): # prueba un tweet nuevo en base aun modelo ya creado
    z = mapTweet(tweet, afinn, emoticonDict, positive, negative, neutral, slangs)
    z_scaled = scaler.transform([z])
    z = normalizer.transform(z_scaled)
    z = z[0].tolist()
    return model.predict([z]).tolist()

# Preprocesamiento de los archivos
stopWords       = preprocessing.getStopWordList(abs_file_url('resources/stopWords.txt'))
slangs          = preprocessing.loadSlangs(abs_file_url('resources/internetSlangs.txt'))
afinn           = polarity.loadAfinn(abs_file_url('resources/afinn.txt'))
emoticonDict    = features.createEmoticonDictionary(abs_file_url('resources/emoticon.txt'))

# Se construye el vector con las palabras más frecuentes presentes en tweets positivos, negativos, y neutrales
positive = ngramGenerator.mostFreqList(abs_file_url('data/used/positive1.csv'), 3000)
negative = ngramGenerator.mostFreqList(abs_file_url('data/used/negative1.csv'), 3000)
neutral  = ngramGenerator.mostFreqList(abs_file_url('data/used/neutral1.csv' ), 3000)

# Normalizamos el tamaño de los unigramas, si es que son menores a 3000
min_len = min([len(positive), len(negative), len(neutral)])

positive = positive[0:min_len]
negative = negative[0:min_len]
neutral  = neutral [0:min_len]

# Cargamos los tweets de entrenamiento
# 4 = positivo
# 2 = neutral
# 0 = negativo
X = [] # features
Y = [] # labels
loadData(X, Y, abs_file_url('data/used/positive1.csv'), '4')
loadData(X, Y, abs_file_url('data/used/neutral1.csv'), '2')
loadData(X, Y, abs_file_url('data/used/negative1.csv'), '0')

#pprint(X)
#pprint(Y)

# estandariza las features
X_scaled = pr.scale(np.array(X))
scaler = pr.StandardScaler().fit(X) # para estandarizar posteriormente los datos de test scaler.transform(X) 

# normaliza las features
X_normalized = pr.normalize(X_scaled, norm='l2') # l2 norm
normalizer = pr.Normalizer().fit(X_scaled)  # para normalizar posteriormente los datos de test normalizer.transform([[-1.,  1., 0.]])

X = X_normalized
X = X.tolist()
# pprint(X)

X_train = np.array(X)
y_train = np.array(Y)

K_OPTIMA = 1
MAX_ACC  = 0.0
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    print ("K = " + str(k))
    print("Accuracy of the model using 5 fold cross validation : %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    if (scores.mean() > MAX_ACC):
        MAX_ACC = scores.mean()
        K_OPTIMA = k

print("\nK OPTIMA = " + str(K_OPTIMA))
print("ACC OPTIMA = %0.4f" % MAX_ACC)

# Creamos el modelo en base a los datos que obtuvimos
clasificador = KNeighborsClassifier(K_OPTIMA)
clasificador.fit(X_train, y_train)

predict = predecir("hello, how are you?", clasificador)
pprint(predict)
predict = predecir("Nice! that was awesome", clasificador)
pprint(predict)
predict = predecir("boo, i dont like it", clasificador)
pprint(predict)