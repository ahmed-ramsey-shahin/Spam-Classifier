# -*- coding: utf-8 -*-
from nltk.stem import PorterStemmer
import numpy as np
import pickle
import string
import json
import re


def get_vocab_list(filename):
    file = open(filename, 'r')
    vocab_list = json.load(file)
    file.close()
    return vocab_list


def extract_features(email, vocab_list):
    x = np.zeros((len(vocab_list), 1))
    for word in email.split(' '):
        if word in vocab_list.keys():
            x[vocab_list[word] - 1] = 1
    return x


def preprocess(email):
    ps = PorterStemmer()
    email = email.lower() # Lowercase all the letters
    email = re.sub(r'<[^>]+>', '', email) # Remove HTML tags
    email = re.sub(r'&.*;', '', email) # Remove &nbsp; tags
    email = re.sub(r'[0-9]+', ' number ', email) # Replace numbers with the word number
    email = re.sub(r'(http|https)://[^\s]*', ' httpaddr ', email) # Replace urls with the word httpaddr
    email = re.sub(r'[^\s]+@[^\s]+', ' emailaddr ', email) # Replace emails with the word emailaddr
    email = re.sub(r'[$]+', ' dollar ', email) # Replace prices with the word dollar
    email = email.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    email = re.sub(r' +', ' ', email) # Remove extra spaces
    email = str.join('', email.splitlines()) # Make the string on one line
    email = email.strip() # Remove whitespaces
    # Word Stemming
    word_list = email.split(' ')
    email = ''
    for word in word_list:
        email = email + ps.stem(word, True) + ' '
    return email


if __name__ == '__main__':
    vocab_list = get_vocab_list('vocab_list.json')
    file = open('email.txt', 'r')
    email = file.read()
    file.close()
    file = open('classifier.obj', 'rb')
    classifier = pickle.load(file)
    file.close()
    email = preprocess(email)
    features = extract_features(email, vocab_list)
    prediction = classifier.predict(np.transpose(features))
    print(prediction)
    print('The email is: {}'.format('spam' if prediction == 1 else 'non-spam'))
