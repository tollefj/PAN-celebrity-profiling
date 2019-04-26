import os
import csv
import pandas as pd
import numpy as np
import glob
# preprocess!
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['rt'])  # remove the retweet tag!

stemmer = SnowballStemmer("english")

import re

import sys
import time

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def merge_sents(sent):
    return ' '.join(sent)


def remove_links_and_html(sentence):
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = re.sub(r'<[^<]+?>', '', sentence)

    return sentence


def remove_punct(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def remove_mentions(sentence):
    # keep the @ to check for mentions among separate groups
    return re.sub(r'@#?\b\w\w+\b', '@', sentence)


def valid_token(tok):
    if '#' in tok:
        # make sure the hashtag is alphanumeric (avoiding arabic etc)
        nohash = tok[1:]
        is_latin = re.sub('[^0-9a-zA-Z]+', '', nohash) == nohash
        if is_latin:
            return tok
        else:
            return ''
    non_stop = tok not in stop_words
    no_rt = 'rt' not in tok
    is_latin = re.sub('[^0-9a-zA-Z]+', '', tok) == tok
    return is_latin and non_stop


def clean_stopwords(sentence):
    tokens = tokenizer.tokenize(sentence)
    return ' '.join([t for t in tokens if valid_token(t)])


def stem(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join([t for t in tokens if valid_token(t)])


def empty_to_nan(sentence):
    if len(sentence) < 1:
        return np.nan
    else:
        return sentence


def clean(tweets):
    s = merge_sents(tweets)
    s = s.lower()
    s = remove_links_and_html(s)
    s = remove_punct(s)
    s = remove_mentions(s)
    s = clean_stopwords(s)
    # finally, make sure we have no empty texts
    s = empty_to_nan(s)
    return s
