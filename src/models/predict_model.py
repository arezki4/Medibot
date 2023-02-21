from imports import *
import features.build_features as features
import models.train_model as trained

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
        bag = bag_of_words(inp, features.words_racine)
        if 1 in bag:
            results = trained.clf.predict([bag])
            results_index = numpy.argmax(results)
            tag = features.labels[results_index]
            responses = "0"
            for tg in features.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else :
            print("I don't understand your question.")
