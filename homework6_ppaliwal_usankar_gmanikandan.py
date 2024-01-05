from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torchtext
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0 #Start of sentence
EOS_token = 1 #End of sentence

class Lang:
    def __init__(self, name):
        self.name = name #Adds the name of the the new name 
        self.word2index = {}  #Create new empty array for word to index dictionary
        self.word2count = {} #Create new empty for word to count for counting the number of words
        self.index2word = {0: "SOS", 1: "EOS"}  #Initializing
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '): #Splitting the sentence to individual words
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: #If this word does not exist as an indexed word yet
            self.word2index[word] = self.n_words #Add this word to the list of words and index it as the 2ndd, 3rd, etc word
            self.word2count[word] = 1 #First time occurence of new word-> count = 1
            self.index2word[self.n_words] = word # Create cross mapping of index to word
            self.n_words += 1 #Now we have 1 more word in our dictionary
        else:
            self.word2count[word] += 1 # Add to the count of already existing word that has been found again

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    ) #Convert to Unicode to ASCII

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip()) #Normalize to ASCII after removing spaces at the beginning and end of the extracted word 
    s = re.sub(r"([.!?])", r" \1", s) #Remove .?!
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) #Remove non-letter, non-punctuation characters
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[1].split(' ')) < MAX_LENGTH and \
        len(p[0].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Question 1
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
print(random.choice(pairs))

# Question 2
# to split the pairs randomly into training and testing sets at a 80:20 split
import random
random.shuffle(pairs)
test_ratio = 0.2
train_pairs = pairs[:int((1- test_ratio)*len(pairs))]
test_pairs = pairs[int((1- test_ratio)*len(pairs)):]

from torchtext.vocab import GloVe, vocab
unk_token = "<unk>"
unk_index = 0
glove_vectors  = GloVe(name='6B', dim=100)
glove_vocab  = vocab(glove_vectors.stoi)
glove_vocab.insert_token("<unk>",unk_index)
glove_vocab.set_default_index(unk_index)
pretrained_embeddings = glove_vectors.vectors
pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_glove, vocab_itos=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if use_glove == False:
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)

        else:
            glove_vectors  = GloVe(name='6B', dim=100)
            glove_vocab  = vocab(glove_vectors.stoi)
            glove_vocab.insert_token("<unk>",unk_index)
            glove_vocab.set_default_index(unk_index)
            pretrained_embeddings = glove_vectors.vectors
            pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
            self.gru = nn.GRU(pretrained_embeddings.shape[1], hidden_size)
          
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


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

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, autoencoder = False, pretrained = False):
    start = time.time()
    print('\nStarting Training Process...\n')
    auto_plot_losses = []
    tran_plot_losses = []
    norm_plot_losses = []
    
    training_pairs = [tensorsFromPair(random.choice(train_pairs))
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()

    if autoencoder:
        print('\nAutoencoder Training...\n')
        auto_loss_total = 0  
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        print_loss_total = 0 # Reset every print_every
        plot_loss_total = 0 # Reset every plot_every

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[0]

            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            auto_loss_total += loss
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                auto_plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        print('\nTranslation Training...\n')
        translation_loss_total = 0  
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=0)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

        print_loss_total = 0 # Reset every print_every
        plot_loss_total = 0 # Reset every plot_every

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            translation_loss_total += loss
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                tran_plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    else: 
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        normal_loss_total = 0
        print_loss_total = 0 # Reset every print_every
        plot_loss_total = 0 # Reset every plot_every

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                    decoder, encoder_optimizer, decoder_optimizer, criterion)
            normal_loss_total += loss
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                norm_plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# for testing the model using the testing set
def test_model(encoder, decoder, max_length = MAX_LENGTH):
    print('\nStarting Testing Process...\n')
    with torch.no_grad():
        total_loss = 0
        count = 1

        testing_pairs = [tensorsFromPair(test_pairs[i])for i in range(len(test_pairs))]
        criterion = nn.NLLLoss()
        
        for i in range(len(testing_pairs)):
            testing_pair = testing_pairs[i]
            input_tensor = testing_pair[0]
            target_tensor = testing_pair[1]
            
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            encoder_hidden = encoder.initHidden()
            
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            
            loss = 0
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
            
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                
                if decoder_input.item() == EOS_token:
                    break
            total_loss += loss
                
    return (total_loss / len(testing_pairs)).item()

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(test_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def __main__():
    # total number of epochs
    T = 25000
    hidden_size = 256
    version1_losses = {}
    print(device)
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, use_glove = False).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # for 20k epochs
    trainIters(encoder1, attn_decoder1, T, print_every=5000, autoencoder = False)
    loss = test_model(encoder1, attn_decoder1)
    print('Loss for 20k epochs:', loss)

    # # for T - 15k epochs
    # trainIters(encoder1, attn_decoder1, T - 15000, print_every=2000, autoencoder = False)
    # loss = test_model(encoder1, attn_decoder1)
    # print('Loss for T - 15k epochs:', loss)
    # version1_losses['T - 15k'] = loss

    # # for T - 10k epochs
    # trainIters(encoder1, attn_decoder1, T - 10000, print_every=3000, autoencoder = False)
    # loss = test_model(encoder1, attn_decoder1)
    # print('Loss for T - 10k epochs:', loss)
    # version1_losses['T - 10k'] = loss

    # # for T - 5k epochs
    # trainIters(encoder1, attn_decoder1, T - 5000, print_every=4000, autoencoder = False)
    # loss = test_model(encoder1, attn_decoder1)
    # print('Loss for T - 5k epochs:', loss)
    # version1_losses['T - 5k'] = loss

    # # for T epochs
    # trainIters(encoder1, attn_decoder1, T, print_every=5000, autoencoder = False)
    # loss = test_model(encoder1, attn_decoder1)
    # print('Loss for T epochs:', loss)
    # version1_losses['T'] = loss

    # hidden_size = 256
    # version2_losses = {}
    # encoder2 = EncoderRNN(input_lang.n_words, hidden_size, use_glove = False).to(device)
    # attn_decoder2 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # # for T - 15k epochs
    # trainIters(encoder2, attn_decoder2, T - 15000, print_every=2000, autoencoder = True)
    # loss = test_model(encoder1, attn_decoder1)
    # version2_losses['T - 15k'] = loss
    # print('Loss for T - 15k epochs:', loss)

    # # for T - 10k epochs
    # trainIters(encoder2, attn_decoder2, T - 10000, print_every=3000, autoencoder = True)
    # loss = test_model(encoder1, attn_decoder1)
    # version2_losses['T - 10k'] = loss
    # print('Loss for T - 10k epochs:', loss)

    # # for T - 5k epochs
    # trainIters(encoder2, attn_decoder2, T - 5000, print_every=4000, autoencoder = True)
    # loss = test_model(encoder1, attn_decoder1)
    # version2_losses['T - 5k'] = loss
    # print('Loss for T - 5k epochs:', loss)

    # # for T epochs
    # trainIters(encoder2, attn_decoder2, T, print_every=5000, autoencoder = True)
    # loss = test_model(encoder1, attn_decoder1)
    # version2_losses['T'] = loss
    # print('Loss for T epochs:', loss)

    # hidden_size = 256
    # version3_losses = []
    # encoder3 = EncoderRNN(input_lang.n_words, hidden_size, use_glove = True).to(device)
    # attn_decoder3 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device) #output_lang.n_words, dropout_p=0.1

    # # for T - 15k epochs
    # trainIters(encoder3, attn_decoder3, T - 15000, print_every=2000, autoencoder = False)
    # loss = test_model(encoder3, attn_decoder3)
    # version2_losses['T - 15k'] = loss
    # print('Loss for T - 15k epochs:', loss)

    # # for T - 10k epochs
    # trainIters(encoder3, attn_decoder3, T - 10000, print_every=3000, autoencoder = False)
    # loss = test_model(encoder3, attn_decoder3)
    # version2_losses['T - 10k'] = loss
    # print('Loss for T - 10k epochs:', loss)

    # # for T - 5k epochs
    # trainIters(encoder3, attn_decoder3, T - 5000, print_every=4000, autoencoder = False)
    # loss = test_model(encoder3, attn_decoder3)
    # version2_losses['T - 5k'] = loss
    # print('Loss for T - 5k epochs:', loss)

    # # for T epochs
    # trainIters(encoder3, attn_decoder3, T - 5000, print_every=4000, autoencoder = False)
    # loss = test_model(encoder3, attn_decoder3)
    # version2_losses['T'] = loss
    # print('Loss for T epochs:', loss)

__main__()