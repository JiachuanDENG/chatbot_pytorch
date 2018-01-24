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
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import data
import seq2seq


SOS_token=0
EOS_token=1
MAX_LENGTH=20
loadModel=True
randomEvaluate=False
USEgreedy=True

parameterPath='../../data/chatbotParameters/parametersV1'
vocab,pairs=data.prepareData('Cornel Corpus',True)

def indexesFromSentence(lang,sentence):
    return [lang.word2index[w] for w in sentence.split(' ')]

def variableFromSentence(lang,sentence):
    indexes=indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    return Variable(torch.LongTensor(indexes).view(-1,1))

def variablesFromPair(pair):
    input_vairable=variableFromSentence(vocab,pair[0])
    output_vairalbe=variableFromSentence(vocab,pair[1])
    return input_vairable,output_vairalbe

def evaluate(encoder,decoder,sentence,greedy=USEgreedy,max_length=MAX_LENGTH):
    input_variable=variableFromSentence(vocab,sentence)
    input_length=input_variable.size()[0]
    encoder_hidden=encoder.initHidden()
    encoder_outputs=Variable(torch.zeros(max_length,encoder.hidden_size))
    
    for ei in range(input_length):
        encoder_output,encoder_hidden=encoder(input_variable[ei],encoder_hidden)
        encoder_outputs[ei]=encoder_output

    if greedy:
        decoder_words=greedyDecode(decoder,encoder_outputs,encoder_hidden)

    else:
        decoder_words=beamDecode(decoder,encoder_outputs,encoder_hidden)
    return decoder_words


class Sentence(object):
    def __init__(self,decoder_hidden,last_idx=SOS_token,sentence_indexs=[],sentence_scores=[]):
        self.last_idx=last_idx
        self.decoder_hidden=decoder_hidden
        self.sentence_indexs=sentence_indexs
        self.sentence_scores=sentence_scores

    def calScore(self,):
        return sum(self.sentence_scores)/(len(self.sentence_scores)*1.)

    def addwords(self,decoder_hidden,topv,topi):
        terminate_sents=[]
        newSentences=[]
        for i in range(len(topv[0])):
            if topi[0][i]==EOS_token:
                terminate_sents.append(([vocab.index2word[idx] for idx in self.sentence_indexs]+['<EOS>'],self.calScore()))
                continue
            cur_indexes=self.sentence_indexs[:]
            cur_scores=self.sentence_scores[:]
            cur_indexes.append(topi[0][i])
            cur_scores.append(topv[0][i])
            newSentences.append(Sentence(decoder_hidden,topi[0][i],cur_indexes,cur_scores))
        #print 'terminates:',terminate_sents
        # print 'newSentences:',newSentences
        return terminate_sents,newSentences
    def index2wordsScore(self,):
        words=[]
        for i in range(len(self.sentence_indexs)):
            if self.sentence_indexs[i]==EOS_token:
                words.append('<EOS>')
            else:
                words.append(vocab.index2word[self.sentence_indexs[i]])
        if self.sentence_indexs[-1]!=EOS_token:
            words.append('<EOS>')
        return (words,self.calScore())





   
def beamDecode(decoder,encoder_outputs,encoder_hidden,beam_size=15,max_length=MAX_LENGTH):
    decoder_hidden=encoder_hidden
    cur_top_sents=[Sentence(decoder_hidden)]
    next_top_sents=[]
    terminates=[]
    for di in range(max_length):
        for sent in cur_top_sents:
            decoder_input=Variable(torch.LongTensor([[sent.last_idx]]))
            decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
            topv,topi=decoder_output.data.topk(beam_size)
            terminate_sents,newSentences=sent.addwords(decoder_hidden,topv,topi)
            terminates.extend(terminate_sents)
            next_top_sents.extend(newSentences)
        next_top_sents.sort(key=lambda sent:sent.calScore(),reverse=True)
        cur_top_sents=next_top_sents[:beam_size]
    terminates.extend([sent.index2wordsScore() for sent in cur_top_sents])
    terminates.sort(key=lambda t: t[1],reverse=True)
    # print terminates
    return terminates[0][0]


def greedyDecode(decoder,encoder_outputs,encoder_hidden,max_length=MAX_LENGTH):
    decoder_input=Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden=encoder_hidden
    
    decoder_words=[]
    decoder_attentions=torch.zeros(max_length,max_length)
    
    for di in range(max_length):
        decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
        decoder_attentions[di]=decoder_attention.data
        topv,topi=decoder_output.data.topk(1)
        ni=topi[0][0]
        if ni==EOS_token:
            decoder_words.append('<EOS>')
            break
        else:
            decoder_words.append(vocab.index2word[ni])
        
        decoder_input=Variable(torch.LongTensor([[ni]]))
    return decoder_words


def evaluateRandomly(encoder,decoder,n=10):
    for i in range(n):
        pair=random.choice(pairs)
        print '>',pair[0]
        print '=',pair[1]
        output_words=evaluate(encoder,decoder,pair[0])

        output_sentence=' '.join(output_words)
        print '<',output_sentence
        print ''  

def evaluateInput(encoder, decoder, voc=vocab):
    pair = ''
    while(1):
        try:
            pair = raw_input('> ')
            if pair == 'q': break
    
            output_words = evaluate(encoder, decoder, pair)
            # print output_words
            output_sentence = ' '.join(output_words)
            print '<', output_sentence 
            
        except KeyError:
            print("Incorrect spelling.")


if __name__ == '__main__':
    
    hidden_size = 256
    encoder1 = seq2seq.EncoderRNN(vocab.n_words, hidden_size)
    attn_decoder1 = seq2seq.AttnDecoderRNN(hidden_size, vocab.n_words,
                                   1, dropout_p=0.1)

    if loadModel:
        encoder1.load_state_dict(torch.load(parameterPath+'/chatbotEnparametersTrain.pkl'))
        attn_decoder1.load_state_dict(torch.load(parameterPath+'/chatbotDeparametersTrain.pkl'))
    if randomEvaluate:
        evaluateRandomly(encoder1,attn_decoder1)
    else:
        evaluateInput(encoder1,attn_decoder1)



