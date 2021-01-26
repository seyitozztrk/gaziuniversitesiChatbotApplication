import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
from nltk.tokenize import word_tokenize
import json


def my_autocorrect(input_word):
    with open('intent_4.json') as f:
        data = json.load(f)
    data = data['intents']

    # data = data[0]
    list_ = []
    for i in data:
        for t in i['patterns']:
            a = word_tokenize(t.lower())
            for j in a:
                list_.append(j)
    V = set(list_)

    word_freq_dict = {}
    word_freq_dict = Counter(list_)

    probs = {}

    Total = sum(word_freq_dict.values())

    for k in word_freq_dict.keys():
        probs[k] = word_freq_dict[k] / Total

    input_word = input_word.lower()
    if input_word in V:
        return input_word
    else:
        similarities = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
        return (output['Word'].iloc[0])


import nltk
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()
# nltk.download('punkt')
import time

import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops

import random
import json
import pickle
from snowballstemmer import TurkishStemmer

turkStem = TurkishStemmer()

# with open('my_intents2.json') as file:
#     data = json.load(file)
#     print("merhabaaaaaaaaa")

with open('intent_4.json', "rb") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []  # 2.part

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)

            words.extend(wrds)  # iki diziyi birleştirir.
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [turkStem.stemWord(w.lower()) for w in words if
             w != "?"]  # tüm pattern'ın içerisindeki kelimeleri parçalayıp attık.
    words = sorted(list(set(words)))

    labels = sorted(labels)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # until 2-part
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [turkStem.stemWord(w.lower()) for w in doc]

        for w in words:

            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    # 3-part
ops.reset_default_graph()
    # tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

    # model.load("model.tflearn")
    # batch_size : girdilerin sayısı
model.fit(training, output, n_epoch=1000, batch_size=90, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [turkStem.stemWord(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


import pyttsx3


def chat(inp):
    #print("Merhaba benim adım Gazi:) \nÜniversitemiz hakkında merak ettiğin bir konu varsa bana sorabilirsin.")
    while True:
        #inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return responses


#         if accurate < 0.5:
#             print("Bu konuyu mu aramıştınız: ", tag )
#             inp2 = input("evetse 1 e hayır ise 2'ye basınız : ")
#             if inp2 == "1":
#                 print(responses)
#             else :
#                 print("Doğru sözcükler ile tekrar deneyin")

#         else:
#             print(responses)
