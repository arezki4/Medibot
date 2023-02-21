import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk.corpus import wordnet

import numpy
import sklearn
import random
import json
import pickle

from sklearn import tree
from sklearn.model_selection import train_test_split

with open('intents.json') as fichier:
    data = json.load(fichier)

#print(data["intents"][0])

words_in_documents = [] #tableaux des mots racines
labels = []
sentences = [] #tableaux des phrases
duplicated_labels = [] #les labels encore dupliquees

for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words_in_documents.extend(wrds)
            sentences.append(wrds)
            duplicated_labels.append(intent["tag"])
            labels.append(intent["tag"])
#print("\n\n\n\n\n\n\n\n words = \n\n\n\n\n\n" + str(words_racine))
#print("\n\n\n\n\n\n\n\n sentences = \n\n\n\n\n\n" + str(sentences))

#avoir la racine d'un mots et comprendre plus de nuance de mots
words_source = words_in_documents
words_racine = [stemmer.stem(w.lower()) for w in words_in_documents if w != "?"]
words_racine = sorted(list(set(words_racine)))
words_source = sorted(list(set(words_source)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
print(sentences)
for x, doc in enumerate(sentences):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]
    #print("\n\n\n\n wrds = \n\n\n\n\n" + str(wrds))
    for w in words_racine:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(duplicated_labels[x])] = 1

    #print("\n\n\n\n\n\n\n\n output_row = \n\n\n\n\n\n" + str(output_row))
    #print("\n\n\n\n\n\n\n\n labels = \n\n\n\n\n\n" + str(labels))
    #print("\n\n\n\n\n\n\n\n duplicated_labels[x] = \n\n\n\n\n\n" + str(duplicated_labels[x]))

    training.append(bag)
    output.append(output_row)
    #print("\n\n\n\n bag = \n\n\n\n\n" + str(bag))
for f in output :
	if 1 in f: 
		print(f)
		print("\n")

training = numpy.array(training)
output = numpy.array(output)

# Séparation des données en conjoint d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(training, output, test_size=0.2)

# Création de l'objet DecisionTreeClassifier
clf = tree.DecisionTreeClassifier(max_depth=44)

# Entraînement du modèle sur les données d'entraînement
clf.fit(X_train, y_train)

# Evaluation du modèle sur les données de test
accuracy = clf.score(X_test, y_test)
print("Précision du modèle : {:.2f}".format(accuracy))


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    #print("\n\n\n\n\n bag =" + str(bag))
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        bag = bag_of_words(inp, words_racine)
        if 1 in bag:
            results = clf.predict([bag])
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            responses = "0"
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else :
            print("I don't understand your question.")

chat()
