#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings("ignore") # just to ingore gensim deprecated numpy conversion warning
import sys
from pathlib import Path
import string
import gensim
import gensim.downloader as api
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


def load_model():
    # fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
    model_path = Path('./fasttext.model').absolute()
    fname = get_tmpfile(model_path)
    # fasttext_model300.save(fname)
    # print('model saved to', fname)
    print('Loading model from {}...'.format(model_path))
    model = KeyedVectors.load(fname)
    print('Model loaded from', fname)
    print('\n')
    return model


def print_single_prediction(pred_text, model, no_preds=15):
    print('Keywords closely related to \"{}\":'.format(pred_text))
    print(model.most_similar(pred_text, topn=no_preds))
    print()


def predict(pred_text, model):
    if pred_text in model.vocab: # Predicting Word in Vocab
        print_single_prediction(pred_text, model)
        return
    else: # Predicting phrase
        print('\"{}\" is not in vocab; splitting into multiple words...\n'.format(pred_text))
        for char in pred_text:
            if char in string.punctuation: # error checking for strings like dui_lawyer
                pred_text = outside_pred.replace(char, ' ')

    split_pred_text = pred_text.split(' ')
    no_pred_words = []
    for word in split_pred_text:
        if word in model.vocab:
            print_single_prediction(word, model)
        else:
            no_pred_words.append(word)

    for word in no_pred_words:
        print('\nNo predictions found for \"{}\". Please try another word.\n'.format(word))


if __name__=='__main__':
    if len(sys.argv) != 2:
        print('Incorrect arguments!')
        print('Example input: ')
        print('python suggest_keywords.py "divorce lawyer"')
        exit()
    else:
        print()
        pred_text = sys.argv[1]
        # print('Enter your keyword: ', end='') for manual input
        # pred_text = str(input())
        model = load_model()
        predict(pred_text, model)
