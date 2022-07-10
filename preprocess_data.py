# -*- coding: utf-8 -*-
from nltk.stem import PorterStemmer
import numpy as np
import string
import os
import re


def read_header_and_content(classes):
    output = []
    for _class in classes:
        data = os.listdir(_class)
        for file_name in data:
            file = open('{}/{}'.format(_class, file_name), 'r')
            flag = False
            content = ''
            for i, line in enumerate(file):
                if line.strip() == '':
                    flag = True
                    continue
                if flag == False:
                    continue
                else:
                    content += line
            result = [content, 1 if _class == 'spam' else 0]
            output.append(result)
            file.close()
    return output


def preprocess(data):
    ps = PorterStemmer()
    for i in data:
        i[0] = i[0].lower() # Lowercase all the letters
        i[0] = re.sub(r'<[^>]+>', '', i[0]) # Remove HTML tags
        i[0] = re.sub(r'&.*;', '', i[0]) # Remove &nbsp; tags
        i[0] = re.sub(r'[0-9]+', ' number ', i[0]) # Replace numbers with the word number
        i[0] = re.sub(r'(http|https)://[^\s]*', ' httpaddr ', i[0]) # Replace urls with the word httpaddr
        i[0] = re.sub(r'[^\s]+@[^\s]+', ' emailaddr ', i[0]) # Replace emails with the word emailaddr
        i[0] = re.sub(r'[$]+', ' dollar ', i[0]) # Replace prices with the word dollar
        i[0] = i[0].translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
        i[0] = re.sub(r' +', ' ', i[0]) # Remove extra spaces
        i[0] = str.join('', i[0].splitlines()) # Make the string on one line
        i[0] = i[0].strip() # Remove whitespaces
        # Word Stemming
        word_list = i[0].split(' ')
        i[0] = ''
        for word in word_list:
            i[0] = i[0] + ps.stem(word, True) + ' '
    return data


def get_vocab_list(data):
    vocab_list = {}
    for i in data:
        for word in i[0].split():
            if word in vocab_list.keys():
                vocab_list[word] = vocab_list[word] + 1;
            else:
                vocab_list[word] = 1
    for k, v in list(vocab_list.items()):
        if v < 100:
            del vocab_list[k]
    for index, value in enumerate(vocab_list.keys()):
        vocab_list[value] = index
    return vocab_list


def extract_features(info, vocab_list):
    x = np.zeros((len(vocab_list), 1))
    for word in info[0].split(' '):
        if word in vocab_list.keys():
            x[vocab_list[word]] = 1
    return x


def extract_dataset(data, vocab_list):
    X = np.zeros((len(data), len(vocab_list) + 1))
    for index, info in enumerate(data):
        x = extract_features(info, vocab_list)
        x = np.transpose(x)
        x = np.append(x, info[1])
        X[index] = x
    return X


if __name__ == '__main__':
    current_path = os.getcwd()
    os.chdir('Spam Assassin Dataset')
    classes = os.listdir()
    data = read_header_and_content(classes)
    data = preprocess(data)
    vocab_list = get_vocab_list(data)
    X = extract_dataset(data, vocab_list)
    # np.save('dataset.npy', X)
    os.chdir(current_path)
    