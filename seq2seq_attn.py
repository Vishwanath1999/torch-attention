import torch as T
import torch as nn
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
    def __init__(self,

class Decoder(nn.Module):
    pass

class Seq2Seq(nn.Module):
    pass
