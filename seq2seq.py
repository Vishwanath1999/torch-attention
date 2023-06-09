import torch 
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
# from torch.utils.tensorboard import SummaryWriter
import wandb
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy.eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger,lower=True,init_token='<sos>',eos_token='<eos>')
english = Field(tokenize=tokenizer_eng,lower=True,init_token='<sos>',eos_token='<eos>')

train_data,valid_data,test_data = Multi30k.splits(exts=('.de','.en'),fields=(german,english))
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
    
num_epochs = 20
lr = 1e-3
batch_size=64

load_model = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)

output_size = len(english.vocab)

encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

train_iterator, valid_iterator,test_iterator = BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size=batch_size,sort_within_batch=True,
    sort_key= lambda x: len(x.src),
    device=device)

encoder_net = Encoder(input_size_encoder,encoder_embedding_size,\
                      hidden_size,num_layers,encoder_dropout).to(device)

decoder_net = Decoder(input_size_decoder,decoder_embedding_size,\
                      hidden_size,num_layers,decoder_dropout).to(device)

model = Seq2Seq(encoder_net,decoder_net).to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)
pad_idx = english.voacb.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    torch.load_checkpoint(torch.load('my_checkpoint.pth.ptar'),model,optimizer)


for epoch in range(num_epochs):
    checkpoint = {'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()}
    
    model.train()
    for idx,batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.target.to(device)

        output = model(inp_data,target)
        # output shape: target_len,batch_size,output_dim

        output_dim = output[1:].reshape(-1,output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        train_loss = criterion(output,target)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()

        # wandb.log(loss.item())
    
    model.eval()

    with torch.no_grad():
        for idx,batch in enumerate(valid_iterator):
            inp_data = batch.src.to(device)
            target = batch.target.to(device)

            output = model(inp_data,target)
            # output shape: target_len,batch_size,output_dim

            output_dim = output[1:].reshape(-1,output.shape[2])
            target = target[1:].reshape(-1)
            val_loss = criterion(output,target)

    


score = bleu(test_data,model,german,english,device)
print('Bleu score', score)