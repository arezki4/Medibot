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
from joblib import dump, load
from sklearn import tree
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import recall_score, precision_score, accuracy_score
