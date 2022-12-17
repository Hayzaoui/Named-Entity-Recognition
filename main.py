from gensim import downloader
import numpy as np
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn import metrics
from FNN import model2
from torch import save
import pickle
GLOVE_PATH = 'glove-twitter-200'
glove = downloader.load(GLOVE_PATH)
stopsymb = [')','(',':','$','!','-','~','/','*','?','"',"'",'..','.','_','=','&','^']

def checkstopsymb(word, stopsymb):
  for l in word:
    if l in stopsymb:
      return True
  return False

import time
def is_hh_mm_time(time_string):
    try:
        time.strptime(time_string, '%H:%M')
    except ValueError:
        return False
    return True


def features(word, word2vec):
    vec = list(word2vec)
    if len(word2vec) != 200:
        print('Error')
        return
    if word[0].isupper():  # if the first letter is a Capital letter
        vec.insert(200, 1)
    else:
        vec.insert(200, 0)

    if word[0] == '@':  # if the first letter is a @
        vec.insert(201, 1)
    else:
        vec.insert(201, 0)

    if (all(ord(l) < 128 for l in word)):  # if all the letter from ASCI
        vec.insert(202, 0)
    else:
        vec.insert(202, 1)

    if word.isnumeric():  # if the word is a number
        vec.insert(203, 1)
    else:
        vec.insert(203, 0)

    if (all(l.isupper() for l in word)):  # if all letter are Capital letters
        vec.insert(204, 1)
    else:
        vec.insert(204, 0)

    if word[0] == '#':  # if the first letter is a @
        vec.insert(205, 1)
    else:
        vec.insert(205, 0)

    if word[0:7] == 'http://':  # if the word begin woth http:// that means is a link
        vec.insert(206, 1)
    else:
        vec.insert(206, 0)

    if word[-1] == '.':  # if the last word is .
        vec.insert(207, 1)
    else:
        vec.insert(207, 0)

    if is_hh_mm_time(word):  # if the word represent time like hh:mm
        vec.insert(208, 1)
    else:
        vec.insert(208, 0)

    if checkstopsymb(word, stopsymb):  # if the word contains a symbol from stopsylb list
        vec.insert(209, 1)
    else:
        vec.insert(209, 0)

    if word[0] in stopsymb:  # if the first letter is a symbol from stopsylb list
        vec.insert(210, 1)
    else:
        vec.insert(210, 0)

    if (all(l in stopsymb for l in word)):  # if all the letter are symbol from stopsylb list
        vec.insert(211, 1)
    else:
        vec.insert(211, 0)

    return np.array(vec)


def trainset(train_path):
    vecs_m2 = []
    vecs_m1 = []

    tags = []
    not_existing = {}
    tags_dict = {}
    i = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for lines in f.readlines():
            if lines.split('\t') != ['', '\n'] and lines.split('\t') != ['\n']:
                line = lines.split('\t')
                realword = line[0]
                word = line[0].lower()
                tag = line[1][0]

                if word not in glove.key_to_index:

                    if word not in not_existing.keys():
                        not_existing[word] = {tag: 1}
                    else:
                        if tag not in not_existing[word].keys():
                            not_existing[word][tag] = 1
                        else:
                            not_existing[word][tag] += 1

                        vecs_m2.append(features(realword,np.zeros(200)))
                        vecs_m1.append(np.zeros(200))
                        if tag == 'O':
                            tags.append(0)
                        else:
                            tags.append(1)
                else:
                    vecs_m2.append(features(realword,glove[word]))
                    vecs_m1.append(glove[word])
                    if tag == 'O':
                        tags.append(0)
                    else:
                        tags.append(1)
    return vecs_m1,vecs_m2, tags, tags_dict

def dev(dev_path):
    tags_dev = []
    vecs_dev_m2 = []
    vecs_dev_m1 = []
    not_existing_dev = {}
    index_unseen = {}
    words_dev = {}
    i = 0
    j = 0
    tags_dict_dev = {}
    with open(dev_path, 'r', encoding='utf-8-sig') as f:
        for lines in f.readlines():
            if lines.split('\t') != ['', '\n'] and lines.split('\t') != ['\n']:
                line = lines.split('\t')
                realword = line[0]
                word = line[0].lower()
                words_dev[i] = word
                if len(line) == 1:
                    break
                tag = line[1][0]
                if word not in glove.key_to_index:
                    index_unseen[i] = word
                    if word not in not_existing_dev.keys():
                        not_existing_dev[word] = {tag: 1}
                    else:
                        if tag not in not_existing_dev[word].keys():
                            not_existing_dev[word][tag] = 1
                        else:
                            not_existing_dev[word][tag] += 1
                        vecs_dev_m2.append(features(realword,np.zeros(200)))
                        vecs_dev_m1.append(np.zeros(200))
                        if tag == 'O':
                            tags_dev.append(0)
                        else:
                            tags_dev.append(1)
                    # print(word + ": Not existing")
                else:
                    vecs_dev_m2.append(features(realword,glove[word]))
                    vecs_dev_m1.append(glove[word])
                    if tag == 'O':
                        tags_dev.append(0)
                    else:
                        tags_dev.append(1)
            i += 1
    return vecs_dev_m1,vecs_dev_m2, tags_dev, tags_dict_dev

def model1(vecs, vecs_dev, tags, tags_dev):
    svm = SVC(C=10)
    svm = svm.fit(vecs, tags)
    predicted = svm.predict(vecs_dev)
    print("Model1 F1_score=",metrics.f1_score(tags_dev, predicted))
    with open('model1.pkl', 'wb') as f:
        pickle.dump(svm, f)

if __name__ == '__main__':
    train_path = 'data/train.tagged'
    dev_path = 'data/dev.tagged'
    vecs_m1, vecs_m2, tags, tags_dict = trainset(train_path)

    vecs_dev_m1, vecs_dev_m2, tags_dev, tags_dict_dev = dev(dev_path)
    model1(vecs_m1, vecs_dev_m1, tags, tags_dev)
    model2(vecs_m2, vecs_dev_m2, tags_dict, tags, tags_dev, tags_dict_dev)

