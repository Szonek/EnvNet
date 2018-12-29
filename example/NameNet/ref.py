from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import example.NameNet.model as model
def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())



import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = letterToTensor('A')
hidden =torch.zeros(1, n_hidden)

output, next_hidden = rnn(input, hidden)

input = lineToTensor('Psikuta')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):
    #top_n, top_i = output.topk(1)
    sort_idx = np.argsort(output)
    category_i = sort_idx[0][0][0][-1]
    print(all_categories[category_i], category_i)
    category_i = sort_idx[0][0][0][-2]
    print(all_categories[category_i], category_i)
    category_i = sort_idx[0][0][0][-3]
    print(all_categories[category_i], category_i)


my_model = model.Model("weights")
out = None
hidden = np.zeros((1, 128))
for i in range(input.shape[0]):
    my_model.set_input(input[i].numpy())
    my_model.set_hidden(hidden)
    out = my_model.execute()
    out, hidden = out["output"], out["i2h"]
print(out)
categoryFromOutput(out)
print('end')
#
# criterion = nn.NLLLoss()
#
# def categoryFromOutput(output):
#     top_n, top_i = output.topk(1)
#     category_i = top_i[0].item()
#     return all_categories[category_i], category_i
#
# print(categoryFromOutput(output))
#
# import random
#
# def randomChoice(l):
#     return l[random.randint(0, len(l) - 1)]
#
# def randomTrainingExample():
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
#     line_tensor = lineToTensor(line)
#     return category, line, category_tensor, line_tensor
#
# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line)
#
#
# learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
#
# def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()
#
#     rnn.zero_grad()
#
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#
#     loss = criterion(output, category_tensor)
#     loss.backward()
#
#     # Add parameters' gradients to their values, multiplied by learning rate
#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data)
#
#     return output, loss.item()
#
#
# import time
# import math
#
# n_iters = 100000
# print_every = 5000
# plot_every = 1000
#
#
#
# # Keep track of losses for plotting
# current_loss = 0
# all_losses = []
#
# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
#
# start = time.time()
#
# for iter in range(1, n_iters + 1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train(category_tensor, line_tensor)
#     current_loss += loss
#
#     # Print iter number, loss, name and guess
#     if iter % print_every == 0:
#         guess, guess_i = categoryFromOutput(output)
#         correct = '✓' if guess == category else '✗ (%s)' % category
#         print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
#
#     # Add current loss avg to list of losses
#     if iter % plot_every == 0:
#         all_losses.append(current_loss / plot_every)
#         current_loss = 0
#
# for n, p in rnn.named_parameters():
#     np.savetxt("weights/" + n, p.data.flatten())
#
