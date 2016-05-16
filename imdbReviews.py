"""
This script is what created the dataset pickled.

output:
    Building dictionary.. 3007414  total words  89260  unique words
"""

dataset_path='/Users/changhxu/LDA/aclImdb/'

import numpy
import cPickle as pkl

from collections import OrderedDict
from nltk.corpus import stopwords

import glob
import os
import re
import string

def extract_words(sentences):
    result = []
    stop = stopwords.words('english')
    trash_characters = '?.,!:;"$%^&*()#@+/0123456789<>=\\[]_~{}|`'
    trans = string.maketrans(trash_characters, ' '*len(trash_characters))

    for text in sentences:
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = text.replace('<br />', ' ')
        text = text.replace('--', ' ').replace('\'s', '')
        text = text.translate(trans)
        text = ' '.join([w for w in text.split() if w not in stop])

        words = []
        for word in text.split():
            word = word.lstrip('-\'\"').rstrip('-\'\"')
            if len(word)>2:
                words.append(word.lower())
        text = ' '.join(words)
        result.append(text.strip())
    return result

def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    sentences = extract_words(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = extract_words(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

def load_data(path="imdb.pkl", nb_words=80000, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):

    f = open(path, 'rb')
    train_set = pkl.load(f)
    test_set = pkl.load(f)
    f.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y
    
    def remove_unk(x):
        return [[1 if w >= nb_words else w for w in sen] for sen in x]

    X_train, y_train = train_set
    X_test, y_test = test_set
    X_train = remove_unk(X_train)
    X_test = remove_unk(X_test)

    return (X_train, y_train), (X_test, y_test)

def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train'))

    train_x_pos = grab_data(path+'train/pos', dictionary)
    train_x_neg = grab_data(path+'train/neg', dictionary)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos', dictionary)
    test_x_neg = grab_data(path+'test/neg', dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    print(len(test_y))
    f = open('imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()
