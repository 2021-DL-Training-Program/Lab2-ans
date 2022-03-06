from __future__ import unicode_literals, print_function, division
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataloader import load
from translate import index2word, word2index
from My_GRU import GRU, GRU2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Max length of word
MAX_LENGTH = 22

#----------Hyper Parameters----------#
# The number of vocabulary ('SOS', 'EOS', 'PAD', 'a', 'b', ..., 'z')
vocab_size = 29

hidden_size = 512
batch_size = 32
teacher_forcing_ratio = 0.5
n_iters = 75000
print_every = 100
plot_every = 100
learning_rate = 0.01

# hyperparameter for learning rate scheduler
milestones = [30000, 50000, 70000]
gamma = 0.5

# build index to word table
index_table = {0:'SOS', 1:'EOS', 2:'PAD'}
for i in range(97, 97+26):
    index_table[i - 94] = chr(i)

# build word to index table
word_table = {}
for i in range(len(index_table)):
    word_table[index_table[i]] = i

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = GRU2(hidden_size, hidden_size)

    def forward(self, input, hidden, batch_size=1):
        embedded = self.embedding(input).view(1, batch_size, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = GRU2(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, batch_size=1):
        output = self.embedding(input).view(1, batch_size, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden
        


           
def train(input_list, target_list, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, batch_size, max_length=vocab_size):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    loss = 0
    
    # initialize the hidden state
    encoder_hidden = encoder.initHidden(batch_size)
        
    input_tensor = torch.tensor(input_list, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_list, dtype=torch.long, device=device)

    #----------sequence to sequence part for encoder----------#
    for di in range(MAX_LENGTH):
        _, encoder_hidden = encoder(input_tensor.T[di], encoder_hidden, batch_size)
        
    decoder_input = torch.tensor([word_table['SOS'] for i in range(batch_size)], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	
    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(1, MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
            loss += criterion(decoder_output, target_tensor.T[di].view(-1))
            decoder_input = target_tensor.T[di].view(1, -1).detach()  # Teacher forcing
           
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(1, MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
            loss += criterion(decoder_output, target_tensor.T[di].view(-1))
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / MAX_LENGTH


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



def trainIters(encoder, decoder, n_iters=75000, print_every=1000, plot_every=100, learning_rate=0.01, teacher_forcing_ratio=1, batch_size=1, milestones=[30000, 40000], gamma=0.5):
    start = time.time()
    plot_losses = []
    plot_bleu_score = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0   # Reset every plot_every
    
    best_loss = 100000
    
    # optimizer and learning rate scheduler
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=milestones, gamma=gamma)
    
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=milestones, gamma=gamma)
    
    # load data
    pairs = load('train.json')
    test_pairs = load('test.json')
    
    # loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    for iter in range(0, n_iters):
        # random select a batch of data pairs
        train_pair = [random.choice(pairs) for i in range(batch_size)]
        
        input_list = []
        target_list = []
        
        # translate word to index
        for input, target in train_pair:
            input_list.append(word2index(input))
            target_list.append(word2index(target))

        loss = train(input_list, target_list, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, batch_size)
        
        with torch.no_grad():
            print_loss_total += loss
            plot_loss_total += loss
            
            if (iter + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (iter + 1)/ n_iters), iter + 1, (iter + 1)/ n_iters * 100, print_loss_avg))
            
            if (iter + 1) % plot_every == 0:
                encoder.eval()
                decoder.eval()
                
                score = 0
                target_list = []
                predict_list = [] 
                for word, target in test_pairs:
                    word = word2index(word)
                    target_list.append(target)
                    pred = predict(encoder, decoder, torch.tensor(word, dtype=torch.long, device=device), torch.tensor(word2index(target), dtype=torch.long, device=device))
                    predict_list.append(pred)
                    score += compute_bleu(pred, target)
                avg_score = score / len(test_pairs)
                plot_bleu_score.append(avg_score)
                print('iter:{}\nBleu-4 score:{}\n'.format(iter + 1, avg_score))
                
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0
                plot_losses.append(plot_loss_avg)
                
                # save_model
                if best_loss > plot_loss_avg:
                    best_loss = plot_loss_avg
                    torch.save(encoder.state_dict(), 'encoder.pth')
                    torch.save(decoder.state_dict(), 'decoder.pth')
                    
        encoder_scheduler.step()
        decoder_scheduler.step()
        
    return plot_losses, plot_bleu_score
    
def predict(encoder, decoder, input_tensor, target_tensor):
    with torch.no_grad():
        word = [word_table['SOS']]
        
        # initialize hidden state
        encoder_hidden = encoder.initHidden()
        
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        #----------sequence to sequence part for encoder----------#
        for di in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[di], encoder_hidden)
            
        #----------sequence to sequence part for decoder----------#
        decoder_input = torch.tensor([word_table['SOS']], device=device)
        decoder_hidden = encoder_hidden
        for di in range(1, target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, True)
            decoder_input = F.softmax(decoder_output,dim=1).argmax(dim=1).view(1, -1).detach()
            word.append(decoder_output.argmax().item())
            if decoder_input.item() == word_table['EOS']:
                break
            
        del decoder_input, decoder_hidden, decoder_output
        torch.cuda.empty_cache()
        
    return index2word(word)
        
def test(encoder, decoder, new=False):
    if new:
        test_pairs = load('new_test.json')
    else:
        test_pairs = load('test.json')
    
    score = 0
    for word, target in test_pairs:
        word = word2index(word)
        pred = predict(encoder, decoder, torch.tensor(word, dtype=torch.long, device=device), torch.tensor(word2index(target), dtype=torch.long, device=device))
        score += compute_bleu(pred, target)
        print('===================')
        print('input:  {}\ntarget: {}\npred:   {}'.format(index2word(word), target, pred))
        
    avg_score = score / len(test_pairs)
    print('Bleu-4 score:{}'.format(avg_score))
        
        
encoder = EncoderRNN(vocab_size, hidden_size, batch_size).to(device)
decoder = DecoderRNN(hidden_size, vocab_size, batch_size).to(device)

#-----------------------test---------------------------#

# load model
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))

test(encoder, decoder, False)

#-----------------------train--------------------------#

# plot_losses, plot_bleu_score = trainIters(encoder, decoder, n_iters, print_every, plot_every, learning_rate, teacher_forcing_ratio, batch_size, milestones, gamma)

# # visualize 
# loss_line, = plt.plot(range(1, n_iters+1, plot_every), plot_losses, label='loss')
# bleu4_line, = plt.plot(range(1, n_iters+1, plot_every), plot_bleu_score, label='bleu4-score')

# plt.xlabel('iteration(s)')
# plt.legend(handles = [loss_line, bleu4_line], loc='upper left', fontsize='small')
# plt.savefig('curve.png')