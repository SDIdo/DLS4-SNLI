import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import json
import numpy as np
from torch.utils.data import DataLoader

TRAIN_PATH = "data/snli_1.0_train.jsonl"
TEST_PATH = "data/snli_1.0_test.jsonl"
DEV_PATH = "data/snli_1.0_dev.jsonl"
GLOVE_PATH = "data/glove.840B.300d.txt"
UNKNOWN = "UNKNOWN"
PAD = "PAD"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 300



class NLIModel(nn.Module):

    def __init__(self, vecs, emb_d=EMBEDDING_DIM, mlp_d=1600, lstm_hidden_dims=[512, 1024, 2048], dropout=0.1):
        super().__init__()
        self.E = nn.Embedding.from_pretrained(vecs)
        self.lstm = nn.LSTM(input_size=emb_d, hidden_size=lstm_hidden_dims[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(emb_d + lstm_hidden_dims[0] * 2), hidden_size=lstm_hidden_dims[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(emb_d + (lstm_hidden_dims[0] + lstm_hidden_dims[1]) * 2), hidden_size=lstm_hidden_dims[2],
                              num_layers=1, bidirectional=True)

        self.mlp_1 = nn.Linear(lstm_hidden_dims[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(dropout),
                                          self.sm])

    def forward(self, x1, x1_l, x2, x2_l):
        """
        Args:
            pre_input (Variable): A long variable containing indices for
                premise words. Size: (max_length, batch_size).
            pre_lengths (Tensor): A long tensor containing lengths for
                sentences in the premise batch.
            hyp_input (Variable): A long variable containing indices for
                hypothesis words. Size: (max_length, batch_size).
            pre_lengths (Tensor): A long tensor containing lengths for
                sentences in the hypothesis batch.
        Returns:
            output (Variable): A float variable containing the
                unnormalized probability for each class
        :return:
        """

        x1_emb = self.E(x1)
        x2_emb = self.E(x2)

        # 1
        x1_packed = pack_padded_sequence(x1_emb, x1_l, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(x2_emb, x2_l, batch_first=True, enforce_sorted=False)

        output_x1, (hn, cn) = self.lstm(x1_packed.float())
        output_x2, (hn, cn) = self.lstm(x2_packed.float())

        output_x1_unpacked, _ = pad_packed_sequence(output_x1, batch_first=True)
        output_x2_unpacked, _ = pad_packed_sequence(output_x2, batch_first=True)


        # Using residual connection
        x1_emb = torch.cat([x1_emb, output_x1_unpacked], dim=2)
        x2_emb = torch.cat([x2_emb, output_x2_unpacked], dim=2)

        # 2
        x1_packed = pack_padded_sequence(x1_emb, x1_l, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(x2_emb, x2_l, batch_first=True, enforce_sorted=False)

        output_x1, (hn, cn) = self.lstm_1(x1_packed.float())
        output_x2, (hn, cn) = self.lstm_1(x2_packed.float())

        output_x1_unpacked, _ = pad_packed_sequence(output_x1, batch_first=True)
        output_x2_unpacked, _ = pad_packed_sequence(output_x2, batch_first=True)

        x1_emb = torch.cat([x1_emb, output_x1_unpacked], dim=2)
        x2_emb = torch.cat([x2_emb, output_x2_unpacked], dim=2)

        # 3
        x1_packed = pack_padded_sequence(x1_emb, x1_l, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(x2_emb, x2_l, batch_first=True, enforce_sorted=False)

        output_x1, (hn, cn) = self.lstm_2(x1_packed.float())
        output_x2, (hn, cn) = self.lstm_2(x2_packed.float())

        x1, _ = pad_packed_sequence(output_x1, batch_first=True)
        x2, _ = pad_packed_sequence(output_x2, batch_first=True)

        x1 = torch.max(x1, 1)[0]
        x2 = torch.max(x2, 1)[0]

        sentence_vector = torch.cat([x1, x2, x1 - x2, x1 * x2], dim=1)
        out = self.classifier(sentence_vector.float())

        return out





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
    words_to_vecs = {}
    index = 0
    indexed_words[PAD] = index
    index += 1
    break1 = 0
    for line in f.readlines():
        splitLines = line.split(" ")
        indexed_words[splitLines[0]] = index
        words_to_vecs[splitLines[0]] = np.asarray(splitLines[1:], dtype='float32')
        index += 1
        break1 += 1
    f.close()
    indexed_words[UNKNOWN] = index

    print("Loading complete!")
    print("Creating Embedding matrix...")
    embedding_matrix = np.random.random((index + 1, EMBEDDING_DIM))
    for word in indexed_words:
        embedding_vector = words_to_vecs.get(word)
        if embedding_vector is not None:
            embedding_matrix[indexed_words[word]] = embedding_vector
    print("Creation complete!")
    return indexed_words, torch.tensor(embedding_matrix)


def create_dataset(premises, labels, hypotheses, indexed_words, indexed_labels):
    print("Creating dataset...")
    dataset = []
    premises_tensors = []
    labels_tensors = []
    hypotheses_tensors = []
    for i in range(len(premises)):
        premise, label, hypothesis = premises[i].replace('.', '').replace('\n', '').split(" "), labels[i], hypotheses[i].replace('.', '').replace('\n', '').split(" ")

        if label == "-":
            label = "entailment"

        for p in range(len(premise)):
            if premise[p] not in indexed_words:
                premise[p] = UNKNOWN

        for h in range(len(hypothesis)):
            if hypothesis[h] not in indexed_words:
                hypothesis[h] = UNKNOWN

        premises_tensors.append(torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in premise]))
        hypotheses_tensors.append(torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in hypothesis]))
        labels_tensors.append(torch.tensor(indexed_labels[label]))

    for p, h, l in zip(premises_tensors, hypotheses_tensors, labels_tensors):
        dataset.append((p, h, l))

    print("Creation complete!")
    return dataset

def pad_collate(batch):
  (x1, x2, l) = zip(*batch)
  x1_lens = [len(p) for p in x1]
  x2_lens = [len(h) for h in x2]

  x1_pad = pad_sequence(x1, batch_first=True, padding_value=0)
  x2_pad = pad_sequence(x2, batch_first=True, padding_value=0)

  return x1_pad, x2_pad, x1_lens, x2_lens, l

def main():
    indexed_words, vecs = create_dict()
    indexed_labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    train_premises, train_labels, train_hypotheses = parse_data(TRAIN_PATH)
    train_dataset = create_dataset(train_premises, train_labels, train_hypotheses, indexed_words, indexed_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    lr = 0.0002


    model = NLIModel(vecs=vecs.to(DEVICE))

    model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.25)  # Need to add decay according to article..
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # loop over the dataset multiple times

        for i, data in enumerate(train_dataloader, 0):
            premise, hypothesis, premise_length, hypothesis_length, label = data
            premise = premise.to(DEVICE)
            hypothesis = hypothesis.to(DEVICE)
            label = torch.tensor(label).to(DEVICE)

            # zero the parameter gradients

            outputs = model(premise, premise_length, hypothesis, hypothesis_length)
            label_pred = outputs.max(1)[1]
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            print("loss is " + str(loss))
            accuracy = torch.eq(label, label_pred).float().mean()
            print("acc is " + str(accuracy))
            loss.backward()
            optimizer.step()
            optimizer.lr = lr * 0.5 if epoch % 2 == 0 else lr


if __name__ == '__main__':
    main()