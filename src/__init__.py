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

import models.predict_model as predict

predict.chat()
