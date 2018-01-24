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
import numpy as np

import data
import seq2seq

SOS_token=0
EOS_token=1
MAX_LENGTH=20
loadModel=True


teacher_forcing_ratio=0.8
parameterPath='../../data/chatbotParameters/parametersV2'
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



def train(input_variable,target_variable,encoder,decoder,encoder_optimizer,\
         decoder_optimizer,criterion,max_length=MAX_LENGTH):
    
    encoder_hidden=encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length=input_variable.size()[0]
    target_length=target_variable.size()[0]
    
    encoder_outputs=Variable(torch.zeros(max_length,encoder.hidden_size))
    
    loss=0.
    
    for ei in range(input_length):
        encoder_output,encoder_hidden=encoder(input_variable[ei],encoder_hidden)
        encoder_outputs[ei]=encoder_output[0][0]
    
    decoder_input=Variable(torch.LongTensor([[SOS_token]]))
    
    decoder_hidden=encoder_hidden
    
    teacher_forcing=True if random.random()<teacher_forcing_ratio else False
    
    if teacher_forcing:
        for di in range(target_length):
            decoder_output,decoder_hidden,attn_weight=decoder(decoder_input,\
                                                          decoder_hidden,\
                                                         encoder_outputs)
            loss+=criterion(decoder_output,target_variable[di])
            decoder_input=target_variable[di]
    else:
        for di in range(target_length):
            decoder_output,decoder_hidden,attn_weight=decoder(decoder_input,\
                                                          decoder_hidden,\
                                                         encoder_outputs)
            loss+=criterion(decoder_output,target_variable[di])
            
            topV,topI=decoder_output.data.topk(1)
            ni=topI[0][0]
            decoder_input=Variable(torch.LongTensor([[ni]]))
            
            if ni==EOS_token:
                break
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0]/target_length
            


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig,ax=plt.subplots()
    loc=ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)       



def trainIters(encoder,decoder,n_iters,print_every=1000,plot_every=100,save_every=1000,learning_rate=0.01):
    start=time.time()
    plot_losses=[]
    print_loss_total=0.
    plot_loss_total=0.
    
    encoder_optimizer=optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer=optim.SGD(decoder.parameters(),lr=learning_rate)
    
    training_pairs=[variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    
    criterion=nn.NLLLoss()
    
    for iter in range(1,n_iters+1):
        training_pair=training_pairs[iter-1]
        loss=train(training_pair[0],training_pair[1],encoder,decoder,encoder_optimizer,\
             decoder_optimizer,criterion)
        print_loss_total+=loss
        plot_loss_total+=loss
        
        if iter%print_every==0:
            print_loss_ave=print_loss_total/print_every
            print_loss_total =0.
            print '{} ({} {}%) {}'.format(timeSince(start,iter/n_iters),\
                                          iter, iter/n_iters*100,print_loss_ave)
        
        if iter%plot_every==0:
            plot_loss_ave=plot_loss_total/plot_every
            plot_loss_total=0.
            plot_losses.append(plot_loss_ave)
        if iter%save_every==0:
				torch.save(decoder.state_dict(),parameterPath+'/chatbotDeparametersTrain.pkl') 
				torch.save(encoder.state_dict(),parameterPath+'/chatbotEnparametersTrain.pkl') 
            
    showPlot(plot_losses)
            
if __name__ == '__main__':
		vocab,pairs=data.prepareData('Cornel Corpus',True)
		hidden_size = 512
		encoder1 = seq2seq.EncoderRNN(vocab.n_words, hidden_size,n_layers=2)
		attn_decoder1 = seq2seq.AttnDecoderRNN(hidden_size, vocab.n_words,2, dropout_p=0.1)

		if loadModel:
			encoder1.load_state_dict(torch.load(parameterPath+'/chatbotEnparametersTrain.pkl'))
			attn_decoder1.load_state_dict(torch.load(parameterPath+'/chatbotDeparametersTrain.pkl'))



		trainIters(encoder1, attn_decoder1, 75000*3, print_every=100)    
		        
		    
		    



