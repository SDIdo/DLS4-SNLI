import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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

        x1_l = x1_l.tolist()
        x2_l = x2_l.tolist()

        x1_packed = pack_padded_sequence(x1_emb, x1_l, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(x2_emb, x2_l, batch_first=True, enforce_sorted=False)

        output_x1, (hn, cn) = self.lstm(x1_packed.float())
        output_x2, (hn, cn) = self.lstm(x2_packed.float())

        output_x1_unpacked, _ = pad_packed_sequence(output_x1, batch_first=True)
        output_x2_unpacked, _ = pad_packed_sequence(output_x2, batch_first=True)

        mat1 = np.zeros((3, 78 - output_x1_unpacked.shape[1], 1024))
        mat2 = np.zeros((3, 58 - output_x2_unpacked.shape[1], 1024))

        mat1 = torch.tensor(mat1).to(DEVICE)
        mat2 = torch.tensor(mat2).to(DEVICE)

        output_x1_unpacked = torch.cat([output_x1_unpacked, mat1], dim=1)
        output_x2_unpacked = torch.cat([output_x2_unpacked, mat2], dim=1)

        # Length truncate
        #len1 = output_x1_unpacked.size(1)
        #len2 = output_x2_unpacked.size(1)
        #x1_emb = x1_emb[:, :len1, :]  # [T, B, D]
        #x2_emb = x2_emb[:, :len2, :]  # [T, B, D]

        # Using residual connection
        x1_emb = torch.cat([x1_emb, output_x1_unpacked], dim=2)
        x2_emb = torch.cat([x2_emb, output_x2_unpacked], dim=2)

        x1_packed = pack_padded_sequence(x1_emb, x1_l, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(x2_emb, x2_l, batch_first=True, enforce_sorted=False)

        output_x1, (hn, cn) = self.lstm_1(x1_packed.float())
        output_x2, (hn, cn) = self.lstm_1(x2_packed.float())

        output_x1_unpacked, _ = pad_packed_sequence(output_x1, batch_first=True)
        output_x2_unpacked, _ = pad_packed_sequence(output_x2, batch_first=True)

        #s1_layer3_in = torch.cat([x1_emb, output_x1_unpacked, s1_layer2_out], dim=2)
        #s2_layer3_in = torch.cat([x2_emb, output_x2_unpacked, s2_layer2_out], dim=2)

        return x1_emb





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
    vecs = []
    index = 0
    indexed_words[PAD] = index
    index += 1
    break1 = 0
    for line in f.readlines():
        if break1 < 1000:
            splitLines = line.split(" ")
            indexed_words[splitLines[0]] = index
            words_to_vecs[splitLines[0]] = np.asarray(splitLines[1:], dtype='float32')
            index += 1
            break1 += 1
        else:
            break
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

        for p in range(len(premise)):
            if premise[p] not in indexed_words:
                premise[p] = UNKNOWN

        for h in range(len(hypothesis)):
            if hypothesis[h] not in indexed_words:
                hypothesis[h] = UNKNOWN

        premises_tensors.append(torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in premise]))
        hypotheses_tensors.append(torch.cat([torch.tensor([indexed_words[w]], dtype=torch.long) for w in hypothesis]))
        labels_tensors.append(torch.tensor(indexed_labels[label]))

    premises_lens = [len(p) for p in premises_tensors]
    hypotheses_lens = [len(h) for h in hypotheses_tensors]
    premises_tensors = nn.utils.rnn.pad_sequence(premises_tensors).transpose(0, 1).tolist()
    hypotheses_tensors = nn.utils.rnn.pad_sequence(hypotheses_tensors).transpose(0, 1).tolist()

    for p, p_l, h, h_l, l in zip(premises_tensors, premises_lens, hypotheses_tensors, hypotheses_lens, labels_tensors):
        dataset.append((p, p_l, h, h_l, l))

    print("Creation complete!")
    return dataset


def main():
    indexed_words, vecs = create_dict()
    indexed_labels = {"-": -1, "entailment": 0, "neutral": 1, "contradiction": 2}
    train_premises, train_labels, train_hypotheses = parse_data(TRAIN_PATH)
    train_dataset = create_dataset(train_premises, train_labels, train_hypotheses, indexed_words, indexed_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    lr = 0.0002


    model = NLIModel(vecs=vecs.to(DEVICE))

    model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.25)  # Need to add decay according to article..
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):  # loop over the dataset multiple times

        for i, data in enumerate(train_dataloader, 0):
            premise, premise_length, hypothesis, hypothesis_length, label = data
            premise = torch.cat(premise).reshape(3, -1)
            premise = premise.to(DEVICE)
            hypothesis = torch.cat(hypothesis).reshape(3, -1)
            hypothesis = hypothesis.to(DEVICE)
            premise_length = premise_length.int()
            hypothesis_length = hypothesis_length.int()
            label = label.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(premise, premise_length, hypothesis, hypothesis_length)
            label_pred = outputs.max(1)[1]
            loss = criterion(outputs, label)
            print("loss is " + str(loss))
            accuracy = torch.eq(label, label_pred).float().mean()
            print("acc is " + str(accuracy))
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()