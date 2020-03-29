import argparse
import datetime
import json
import math
import os
import pandas as pd
import random
import string
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from io import open
from torch.utils.data import DataLoader

from Datasets.StreetDataset import StreetDataset
from Models.LSTM import LSTM
from Utilities.Convert import *
from Utilities.Utilities import plot_losses, timeSince

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='first', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--embed_dim', help='Size of embedding dimension', nargs='?', default=32, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.0005, type=float)
parser.add_argument('--epoch', help='Number of epochs', nargs='?', default=10000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/Addresses.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every', nargs='?', default=50, type=int)
parser.add_argument('--batch', help='Batch size', nargs='?', default=5000, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=0, type=int)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
EPOCH = args.epoch
NUM_LAYERS = args.num_layers
EMBED_DIM = args.embed_dim
LR = args.lr
HIDDEN_SZ = args.hidden_size
TRAIN_FILE = args.train_file
BATCH_SZ = args.batch
COLUMN = args.column
PRINTS = args.print
CLIP = 1

# Global variables
SOS = 'SOS'
PAD = 'PAD'
EOS = 'EOS'
IN_CHARS = [char for char in string.ascii_letters] + ['\'', '-'] + [EOS, SOS, PAD]
IN_COUNT = len(IN_CHARS)
OUT_CHARS = [char for char in string.ascii_letters] + ['\'', '-'] + [EOS, PAD]
OUT_COUNT = len(OUT_CHARS)
MAX_LENGTH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(x: list):
    optimizer.zero_grad()
    loss = 0.

    batch_sz = len(x)
    street_max_len = len(max(x[0], key=len)) + 1  # +1 for EOS xor SOS

    src_x = list(map(lambda s: [SOS] + [char for char in s] + [PAD] * ((street_max_len - len(s)) - 1), x[0]))
    trg_x = list(map(lambda s: [char for char in s] + [EOS] + [PAD] * ((street_max_len - len(s)) - 1), x[0]))

    src = indexTensor(src_x, street_max_len, IN_CHARS).to(DEVICE)
    zip = zipcodeTensor(x[1], 5).to(DEVICE)
    trg = targetsTensor(trg_x, street_max_len, OUT_CHARS).to(DEVICE)

    names = [''] * batch_sz
    
    lstm_hidden = None
    for i in range(src.shape[0]):
        lstm_input = src[i]
        lstm_probs, lstm_hidden = lstm(lstm_input, zip, lstm_hidden)
        best_index = torch.argmax(lstm_probs, dim=2)

        loss += criterion(lstm_probs[0], trg[i])

        for idx in range(len(names)):
            names[idx] += OUT_CHARS[best_index[0][idx].item()]

    loss.backward()
    optimizer.step()

    return names, loss.item()


def iter_train(dl: DataLoader, epoch: int = EPOCH, path: str = "Checkpoints/",
               print_every: int = PRINTS):
    all_losses = []
    total_loss = 0

    for iter in range(1, epoch + 1):
        for x in dl:
            name, loss = train(x)
            total_loss += loss

            if iter % print_every == 0:
                all_losses.append(total_loss / print_every)
                total_loss = 0
                plot_losses(all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss", filename=NAME)
                torch.save({'weights': lstm.state_dict()}, os.path.join(f"{path}{NAME}.path.tar"))


def iter_train_dl(dl: DataLoader, epochs: int = EPOCH, path: str = "Checkpoints/", print_every: int = PRINTS):
    all_losses = []
    total_loss = 0

    for iter in range(1, epochs + 1):
        for x in dl:
            name, loss = train(x)
            total_loss += loss

            if iter % print_every == 0:
                all_losses.append(total_loss / print_every)
                total_loss = 0
                plot_losses(all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss",
                            filename=NAME)
                torch.save({'weights': lstm.state_dict()}, os.path.join(f"{path}{NAME}.path.tar"))


def sample(length: list):
    with torch.no_grad():
        max_length = length[0]
        lstm_input = indexTensor([[SOS]], 1, IN_CHARS).to(DEVICE)
        lng_input = lengthTestTensor([length]).to(DEVICE)
        lstm_hidden = lstm.initHidden(1)
        lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
        name = ''
        char = SOS

        for i in range(max_length):
            lstm_probs, lstm_hidden = lstm(lstm_input[0], lng_input, lstm_hidden)
            lstm_probs = torch.softmax(lstm_probs, dim=2)
            sample = torch.distributions.categorical.Categorical(lstm_probs).sample()
            sample = sample[0]
            char = OUT_CHARS[sample]

            if sample == OUT_CHARS.index(EOS):
                break

            name += char
            lstm_input = indexTensor([[char]], 1, IN_CHARS).to(DEVICE)

        return name


def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)


def save_json(jsonpath: str, content):
    with open(jsonpath, 'w') as jsonfile:
        json.dump(content, jsonfile)


to_save = {
    'session_name': NAME,
    'hidden_size': HIDDEN_SZ,
    'num_layers': NUM_LAYERS,
    'embed_dim': EMBED_DIM,
    'input': IN_CHARS,
    'output': OUT_CHARS,
    'input_sz': IN_COUNT,
    'output_sz': OUT_COUNT,
    'EOS': EOS,
    'SOS': SOS,
    'PAD': PAD,
}

save_json(f'Config/{NAME}.json', to_save)

lstm = LSTM(IN_COUNT, HIDDEN_SZ, OUT_COUNT, padding_idx=IN_CHARS.index(PAD), num_layers=NUM_LAYERS,
               embed_size=EMBED_DIM)

if args.continue_training == 1:
    lstm.load_state_dict(torch.load(f'Checkpoints/{NAME}.path.tar')['weights'])
    
lstm.to(DEVICE)

criterion = nn.NLLLoss(ignore_index=OUT_CHARS.index(PAD))
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

df = pd.read_csv(TRAIN_FILE)
ds = StreetDataset(df)
dl = DataLoader(ds, batch_size=200)
iter_train(dl)