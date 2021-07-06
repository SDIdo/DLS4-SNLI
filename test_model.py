import argparse
import logging
import os
from sys import argv
import numpy as np
import torch
from torchtext import data, datasets
import json
from main import NLIModel
from torch.utils.data import DataLoader
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORDS = 4000001
DIMENSIONS = 300
UNKNOWN = "UNKNOWN"

class TestArgs():
    def __init__(self, model_path, indexed_words_path, test_path='data/snli_1.0_test.jsonl'):
        self.test_path = test_path
        self.model_path = model_path
        self.indexed_words_path = indexed_words_path

def create_random_embedding(words, dimension):
    return torch.rand(words, dimension)

def create_test_set(premises, hypotheses, indexed_words):
    print("Creating dataset...")
    dataset = []
    premises_tensors = []
    hypotheses_tensors = []
    for i in range(len(premises)):
        premise, hypothesis = premises[i].replace('.', '').replace('\n', '').split(" "), hypotheses[
            i].replace('.', '').replace('\n', '').split(" ")

        for p in range(len(premise)):
            if premise[p] not in indexed_words:
                premise[p] = UNKNOWN

        for h in range(len(hypothesis)):
            if hypothesis[h] not in indexed_words:
                hypothesis[h] = UNKNOWN

        premises_tensors.append(torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in premise]))
        hypotheses_tensors.append(torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in hypothesis]))

    premises_lens = [len(p) for p in premises_tensors]
    hypotheses_lens = [len(h) for h in hypotheses_tensors]
    premises_tensors = nn.utils.rnn.pad_sequence(premises_tensors).transpose(0, 1).tolist()
    hypotheses_tensors = nn.utils.rnn.pad_sequence(hypotheses_tensors).transpose(0, 1).tolist()

    for p, p_l, h, h_l, in zip(premises_tensors, premises_lens, hypotheses_tensors, hypotheses_lens):
        dataset.append((p, p_l, h, h_l))

    print("Creation complete!")
    return dataset

def parse_test_data(file):
    premises = []
    hypotheses = []

    print("Loading test data...")
    with open(file) as f:
        for line in f:
            parse_line = json.loads(line)
            premises.append(parse_line["sentence1"])
            hypotheses.append(parse_line["sentence2"])
    print("Loading complete!")
    return premises, hypotheses

def test(args):
    # lstm_hidden_dims = [int(d) for d in args.lstm_hidden_dims.split(',')]
    print('Building model...')
    vecs = create_random_embedding(NUM_WORDS, DIMENSIONS) # TODO ok to say 400k words?
    model = NLIModel(vecs=vecs) # TODO trying to save the cost of loading actual vecs from json.. hoping in the load_state
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda(DEVICE)

    print('Testing model...')
    indexed_labels_rev = {-1: 'not sure', 0: "entailment", 1: "neutral", 2: "contradiction"}
    indexed_words = (open(args.indexed_words_path, "r")).read()
    'indexed_words.json'
    test_premises, test_hypotheses = parse_test_data(args.test_path)
    test_dataset = create_test_set(test_premises, test_hypotheses, indexed_words)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    preds = []
    for batch in test_dataloader:
        pre_input, pre_lengths = batch[0], batch[1]
        hyp_input, hyp_lengths = batch[2], batch[3]
        model_output = model(pre_input, pre_lengths, hyp_input, hyp_lengths)
        label_pred = model_output.max(1)[1]
        preds.append([pre_input, hyp_input, indexed_labels_rev[label_pred]])

    with open('test.pred', "w") as f:
        for pred in preds:
            f.write(str(pred))

if __name__ == "__main__":
    args = TestArgs(*argv[1:])
    test(args)

