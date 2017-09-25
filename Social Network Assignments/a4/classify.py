"""
classify.py
"""
# Imports you'll need.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import os
import time
from TwitterAPI import TwitterAPI
import configparser
from operator import itemgetter, attrgetter
from collections import Counter, defaultdict
import copy
import pickle
from itertools import product
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re

def read_pickle_dump(filename):
    
    file = open(filename,'rb')
    fl = pickle.load(file)
    file.close()

    #print(fl[:2])
    return(fl)

    pass


def get_afinn():

    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    print('read %d AFINN terms.\nE.g.: %s' % (len(afinn), str(list(afinn.items())[:10])))
    
    return(afinn)

    pass

def afinn_sentiment(terms, afinn):
    total = 0.
    for t in terms:
        if t in afinn:
            print('\t%s=%d' % (t, afinn[t]))
            total += afinn[t]
    return total

# Distinguish neutral from pos/neg.
# Return two scores per document.
def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg

# Tokenize tweets
# \w = [a-zA-Z0-9_]
"""
def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()
"""

def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):
   
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens



def unique_top_positive(positives):

    # Print top positives:
    print('Top Positive Tweets ->')
    for tweet, pos, neg in sorted(set(positives), key=lambda x: x[1], reverse=True)[:5]:
        print('(pos,neg) = (%d,%d) -> Tweet = %s'%(pos, neg, tweet))
    
        tokens = tokenize(tweet, lowercase=True, keep_punctuation=False,
                           prefix=None,
                           collapse_urls=True, collapse_mentions=True) 
        #print('Tokens = ',tokens)
    
    pass

def unique_top_negative(negatives):

    print('Top Negative Tweets ->')
    for tweet, pos, neg in sorted(set(negatives), key=lambda x: x[2], reverse=True)[:5]:
        print('(pos,neg) = (%d,%d) -> Tweet = %s'%(pos, neg, tweet))
    
        tokens = tokenize(tweet, lowercase=True, keep_punctuation=False,
                           prefix=None,
                           collapse_urls=True, collapse_mentions=True) 
        #print('Tokens = ',tokens)
        
    pass

def unique_top_neutral(neutral):

    print('Top Neutral Tweets ->')
    for tweet, pos, neg in sorted(set(neutral), key=lambda x: x[2], reverse=True)[:5]:
        print('(pos,neg) = (%d,%d) -> Tweet = %s'%(pos, neg, tweet))
    
        tokens = tokenize(tweet, lowercase=True, keep_punctuation=False,
                           prefix=None,
                           collapse_urls=True, collapse_mentions=True) 
        #print('Tokens = ',tokens)
        
    pass


def classification(tweets) :
   
    # afinn 
    afinn = get_afinn()
    print('AFINN Size = ',len(afinn))

    # get tweets['text'] and remove unwanted part -> tokenize

    #tokens = [tokenize(t['text']) for t in tweets]

    
    tokens = [ tokenize(t['text'], lowercase=True, keep_punctuation=False,
               prefix=None,
               collapse_urls=True, collapse_mentions=True) 
               for t in tweets ]
  
    # find poisitive - negative -  neutral
    positives = []
    negatives = []
    neutral   = []

    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment2(token_list, afinn)
        if pos > neg:
           positives.append((tweet['text'], pos, neg))
        elif neg > pos:
           negatives.append((tweet['text'], pos, neg))
        elif neg == pos:
           neutral.append((tweet['text'], pos, neg))

    unique_top_positive(positives)
    unique_top_negative(negatives)
    unique_top_neutral(neutral)

    #store classes in dict
    
    classes = {}

    classes['positive'] = positives
    classes['negative'] = negatives
    classes['neutral'] = neutral

    return(classes)

    pass

def get_classes_Info(classes):
   
    print('Number of Classes = ',len(classes))
  
    for clas in classes.keys():

        if clas == 'positive' :
           print('Size of positive class = ',len(classes['positive']))

        if clas == 'negative' :
           print('Size of negative class = ',len(classes['negative']))

        if clas == 'neutral' :
           print('Size of neutral class = ',len(classes['neutral']))

    pass

def create_pickle_dump(classes):
    print('Creating Dump File')

    pickle.dump(classes, open('Classes.pkl', 'wb'))

    print('Pickle Dump Created Successfully!!')

    pass


def main():
    """ Main method. You should not modify this. """
   
    # read dump
    filename = 'Tweets.pkl' 
    tweets = read_pickle_dump(filename)
     
    # classification
    classes = classification(tweets) 
    
    # create dump for classes
    create_pickle_dump(classes) 

    get_classes_Info(classes)

    print('Classification Done Successfully!!')

if __name__ == '__main__':
    main()
