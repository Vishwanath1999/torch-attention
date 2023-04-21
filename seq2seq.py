import torch 
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy.eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger,lower=True,init_token='<sos>',eos_token='<eos>')
english = Field(tokenize=tokenizer_eng,lower=True,init_token='<sos>',eos_token='<eos>')

train_data,validation_data,test_data = Multi30k.splits(exts('.de','.en'),fields=(german,english))
german.build_vocab(train_data,max_size=int(1e4),min_freq=2)
english.build_vocab(train_data,max_size=int(1e4),min_freq=2)

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,p):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
    
    def forward(self,x):
        # x shape: (seq_len,N)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden,cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,p):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
        self.fc = nn.Linear(hidden_size,output_size)
    
    def forward(self,x,hidden,cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden,cell) = self.rnn(embedding,(hidden,cell))
        pred = self.fc(outputs)
        pred = pred.squeeze(0)
        return pred,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,source,target,teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len,batch_size,target_vocab_size).to(device)

        hidden,cell = self.encoder(source)
        x = target[0]

        for t in range(1,target_len):
            output,hidden,cell = self.decoder(x,hidden,cell)
            outputs[t] = output
            best_guess = output.argmax(t)
            x = target[t] if random.random()<teacher_force_ratio else best_guess
        
        return outputs

