import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader

TRAIN_PATH = "data/snli_1.0_train.jsonl"
TEST_PATH = "data/snli_1.0_test.jsonl"
DEV_PATH = "data/snli_1.0_dev.jsonl"
GLOVE_PATH = "data/glove.840B.300d.txt"
UNKNOWN = "UNKNOWN"


def parse_data(file):
    premises = []
    labels = []
    hypotheses = []

    print("Loading file data...")
    with open(file) as f:
        for line in f:
            parse_line = json.loads(line)
            premises.append(parse_line["sentence1"])
            labels.append(parse_line["gold_label"])
            hypotheses.append(parse_line["sentence2"])
    print("Loading complete!")
    return premises, labels, hypotheses


def create_dict():
    print("Loading Glove Model")
    f = open(GLOVE_PATH, 'r', encoding="utf8")
    indexed_words = {}
    vecs = []
    index = 0
    for line in f.readlines():
        splitLines = line.split(" ")
        indexed_words[splitLines[0]] = index
        index += 1
        vecs.append(np.array([float(value) for value in splitLines[1:]]))

    indexed_words[UNKNOWN] = index

    print("Loading complete!")
    return indexed_words, np.stack(vecs)


def create_dataset(premises, labels, hypotheses, indexed_words, indexed_labels):
    print("Creating dataset...")
    dataset = []
    for i in range(len(premises)):
        premise, label, hypothesis = premises[i].replace('.', '').split(" "), labels[i], hypotheses[i].split(" ")

        for p in range(len(premise)):
            if premise[p] not in indexed_words:
                premise[p] = UNKNOWN

        for h in range(len(hypothesis)):
            if hypothesis[h] not in indexed_words:
                hypothesis[h] = UNKNOWN

        premise = torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in premise])
        hypothesis = torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in hypothesis])
        label = indexed_labels[label]
        dataset.append((premise, hypothesis, label))
    print("Creation complete!")
    return dataset


# TODO: change to the implementation of the authors
class SentenceEncoder(nn.Module):
    def __init__(self, pretrained_vecs, lstm_hidden_size, linear_hidden_size, output_size, lstm_layers):
        super().__init__()

        # Using pretrained vectors to initialize the embedding
        self.embeddings = nn.Embedding.from_pretrained(pretrained_vecs)

        # Creating a biLSTM with lstm_layers number of layers
        self.lstm = nn.LSTM(input_size=embeddings_vec_size, hidden_size=lstm_hidden_size, bidirectional=True, num_layers=lstm_layers)
        self.fc1 = nn.Linear(in_features=lstm_hidden_size, out_features=linear_hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(in_features=linear_hidden_size, out_features=output_size)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.activation = nn.ReLU()

    def forward(self, sequence):
        embeds = self.embeddings(sequence)
        lstm_input = embeds.view(len(sequence), 1, -1)
        _, (hn, _) = self.lstm(lstm_input)
        output = self.fc1(hn[0])
        output = self.activation(output)
        output = self.fc2(output)

        return output


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        mlp_layers = []
        for i in range(num_layers):
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = hidden_dim
            linear_layer = nn.Linear(in_features=layer_input_dim,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            dropout_layer = nn.Dropout(dropout_prob)
            mlp_layer = nn.Sequential(linear_layer, relu_layer, dropout_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, input):
        """
        Args:
            input (Variable): A float variable of size
                (batch_size, input_dim).
        Returns:
            output (Variable): A float variable of size
                (batch_size, hidden_dim), which is the result of
                applying MLP to the input argument.
        """

        return self.mlp(input)


class NLIClassifier(nn.Module):

    def __init__(self, sentence_dim, hidden_dim, num_layers, num_classes,
                 dropout_prob):
        super().__init__()
        self.sentence_dim = sentence_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        self.mlp = MLP(input_dim=4 * sentence_dim, hidden_dim=hidden_dim,
                       num_layers=num_layers, dropout_prob=dropout_prob)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)

    def forward(self, pre, hyp):
        mlp_input = torch.cat([pre, hyp, (pre - hyp).abs(), pre * hyp], dim=1)
        mlp_output = self.mlp(mlp_input)
        output = self.clf_linear(mlp_output)
        return output


def main():
    indexed_words, vecs = create_dict()
    indexed_labels = {"-": -1, "entailment": 0, "neutral": 1, "contradiction": 2}
    train_premises, train_labels, train_hypotheses = parse_data(TRAIN_PATH)
    train_dataset = create_dataset(train_premises, train_labels, train_hypotheses, indexed_words, indexed_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(train_dataset)


if __name__ == '__main__':
    main()