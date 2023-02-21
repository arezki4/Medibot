from imports import *
with open('data/intents.json') as fichier:
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

            if intent["tag"] not in labels:
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

