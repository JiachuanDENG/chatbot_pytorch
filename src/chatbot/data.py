from __future__ import unicode_literals, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
SOS_token=0
EOS_token=1
MAX_LENGTH=15

def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s=unicode2Ascii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    return s


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 
#         p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

class Vocab(object):
    def __init__(self,name):
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word={0:"SOS",1:"EOS"}
        self.n_words=2
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word]=1
            self.index2word[self.n_words]=word
            self.n_words+=1
        else:
            self.word2count[word]+=1
        
def readQA(name,reverse=False):
    print 'reading lines...'

    lines=open('../../data/processed/QA.txt',encoding='ISO-8859-2').read().strip().split('\n')
    pairs=[[normalizeString(s) for s in l.split('|!|')] for l in lines]
    vocab=Vocab(name)
    if reverse:
        pairs=[list(reversed(p))for p in pairs]
    
    return vocab,pairs


def prepareData(name,reverse=False):
    vocab,pairs=readQA(name,reverse)
    pairs=filterPairs(pairs)
    print 'Read {} sentnece pairs'.format(len(pairs))
    print 'count words...'

    for pair in pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    print 'Counted words:'
    print vocab.name,vocab.n_words
#     print output_lang.name,output_lang.n_words
    return vocab,pairs





