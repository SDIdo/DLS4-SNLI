import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from torch.utils.data import DataLoader
from sys import argv
import numpy as np
import matplotlib.pyplot as plt


UNKNOWN = "UNKNOWN"
PAD = "PAD"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 50 # 300


class NLIModel(nn.Module):

    def __init__(self, vecs, emb_d=EMBEDDING_DIM, mlp_d=1600, lstm_hidden_dims=[512, 1024, 2048], dropout=0.1):
        super().__init__()
        self.E = nn.Embedding.from_pretrained(vecs)
        self.lstm = nn.LSTM(input_size=emb_d, hidden_size=lstm_hidden_dims[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(emb_d + lstm_hidden_dims[0] * 2), hidden_size=lstm_hidden_dims[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(emb_d + (lstm_hidden_dims[0] + lstm_hidden_dims[1]) * 2),
                              hidden_size=lstm_hidden_dims[2],
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

        # 1
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

        mat1 = np.zeros((3, 78 - output_x1_unpacked.shape[1], 2048))
        mat2 = np.zeros((3, 58 - output_x2_unpacked.shape[1], 2048))

        mat1 = torch.tensor(mat1).to(DEVICE)
        mat2 = torch.tensor(mat2).to(DEVICE)

        output_x1_unpacked = torch.cat([output_x1_unpacked, mat1], dim=1)
        output_x2_unpacked = torch.cat([output_x2_unpacked, mat2], dim=1)

        x1_emb = torch.cat([x1_emb, output_x1_unpacked], dim=2)
        x2_emb = torch.cat([x2_emb, output_x2_unpacked], dim=2)

        # 3
        x1_packed = pack_padded_sequence(x1_emb, x1_l, batch_first=True, enforce_sorted=False)
        x2_packed = pack_padded_sequence(x2_emb, x2_l, batch_first=True, enforce_sorted=False)

        output_x1, (hn, cn) = self.lstm_2(x1_packed.float())
        output_x2, (hn, cn) = self.lstm_2(x2_packed.float())

        output_x1_unpacked, _ = pad_packed_sequence(output_x1, batch_first=True)
        output_x2_unpacked, _ = pad_packed_sequence(output_x2, batch_first=True)

        mat1 = np.zeros((3, 78 - output_x1_unpacked.shape[1], 4096))
        mat2 = np.zeros((3, 58 - output_x2_unpacked.shape[1], 4096))

        mat1 = torch.tensor(mat1).to(DEVICE)
        mat2 = torch.tensor(mat2).to(DEVICE)

        x1 = torch.cat([output_x1_unpacked, mat1], dim=1)
        x2 = torch.cat([output_x2_unpacked, mat2], dim=1)

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


def create_dict(glove_path):
    print("Loading Glove Model")
    f = open(glove_path, 'r', encoding="utf8")
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
        premise, label, hypothesis = premises[i].replace('.', '').replace('\n', '').split(" "), labels[i], hypotheses[
            i].replace('.', '').replace('\n', '').split(" ")

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

class Args():
    def __init__(self, train_path="data/snli_1.0_train.jsonl",
                 dev_path="data/snli_1.0_dev.jsonl",
                 test_path="data/snli_1.0_test.jsonl",
                 gloves_path="data/glove.6B.50d.txt", # data/glove.840B.300d.txt
                 val_per_sents=100):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.gloves_path = gloves_path
        self.val_per_sents = val_per_sents

class HyperParameters():
    def __init__(self, lr=0.0002, optimizer='Adam', loss_function='Cross_Entropy', epochs=4, batch_size=32):
        self.lr = lr
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs
        self.batch_size = batch_size

def iterate(lr, model, batch, hyparams, is_training=True):
    batch_size = hyparams.batch_size
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr) if hyparams.optimizer == 'Adam' \
        else torch.optim.Adagrad(model.parameters(), lr=hyparams.lr)
    # Check the lambda implementation for lr decay
    criterion = torch.nn.CrossEntropyLoss() if hyparams.loss_function == 'Cross_Entropy' else torch.nn.MSELoss()
    premise, premise_length, hypothesis, hypothesis_length, label = batch
    premise = torch.cat(premise).reshape(batch_size, -1)
    premise = premise.to(DEVICE)
    hypothesis = torch.cat(hypothesis).reshape(batch_size, -1)
    hypothesis = hypothesis.to(DEVICE)
    premise_length = premise_length.int()
    hypothesis_length = hypothesis_length.int()
    label = label.to(DEVICE)
    # zero the parameter gradients
    outputs = model(premise, premise_length, hypothesis, hypothesis_length)
    label_pred = outputs.max(1)[1]
    loss = criterion(outputs, label)
    accuracy = torch.eq(label, label_pred).float().mean()
    if is_training:
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return loss, accuracy

def export_epochs_train_dev_something_graph(epochs, train_something, dev_something, something, fig_name):
    TRAIN_COLOR = 'green'
    DEV_COLOR = 'orange'
    label_1 = f'train_{something}'
    label_2 = f'dev_{something}'
    plt.title(fig_name)
    plt.plot(epochs, train_something, TRAIN_COLOR, label=label_1)
    plt.plot(epochs, dev_something, DEV_COLOR, label=label_2)
    plt.xlabel('Epochs')
    plt.ylabel(f'{something}')
    plt.legend([label_1, label_2], loc="lower right")
    plt.savefig(f'{fig_name}.jpg')
    plt.clf()

def main():
    if len(argv)>1:
        args = Args(*argv[1:])
    else:
        args=Args()
    hyparams = HyperParameters(lr=0.0002, optimizer='Adam', loss_function='Cross_Entropy'
                               ,epochs=4, batch_size=32)
    batch_size = hyparams.batch_size
    indexed_words, vecs = create_dict(args.gloves_path)
    indexed_labels = {"-": -1, "entailment": 0, "neutral": 1, "contradiction": 2}
    train_premises, train_labels, train_hypotheses = parse_data(args.train_path)
    dev_premises, dev_labels, dev_hypotheses = parse_data(args.dev_path)
    # test_premises, test_labels, test_hypotheses = parse_data(args.test_path)
    train_dataset = create_dataset(train_premises, train_labels, train_hypotheses, indexed_words, indexed_labels)
    dev_dataset = create_dataset(dev_premises, dev_labels, dev_hypotheses, indexed_words, indexed_labels)
    # test_dataset = create_dataset(test_premises, test_labels, test_hypotheses, indexed_words, indexed_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NLIModel(vecs=vecs.to(DEVICE))

    model.to(DEVICE)

    lr = hyparams.lr
    train_loss_lst = []
    train_acc_lst = []
    dev_loss_lst = []
    dev_acc_lst = []
    for epoch in range(hyparams.epochs):  # loop over the dataset multiple times
        lr = lr * 0.5 if epoch % 2 == 0 else lr
        for i, data in enumerate(train_dataloader, 0):
            train_loss, train_accuracy = iterate(lr=lr, model=model, batch=data, hyparams=hyparams)
            print("train acc is " + str(train_accuracy))
            print("train loss is " + str(train_loss))
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_accuracy)

            if i == args.val_per_sents: # per 100 batchs validate model
                num_valid_batches = len(dev_dataloader)
                valid_loss_sum = valid_accracy_sum = 0
                for valid_batch in dev_dataloader:
                    valid_loss, valid_accuracy = iterate(lr=lr, model=model, batch=valid_batch, hyparams=hyparams, is_training=False)
                    valid_loss_sum += valid_loss.data[0]  # Not sure here...
                    valid_accracy_sum += valid_accuracy.data[0]
                dev_loss = valid_loss_sum / num_valid_batches
                dev_accuracy = valid_accracy_sum / num_valid_batches
                print("valid_accuracy is " + str(dev_accuracy))
                print("valid_loss is " + str(dev_loss))
                dev_acc_lst.append(dev_accuracy)
                dev_loss_lst.append(dev_loss)
    # done epochs save the model
    model_filename = (f'epochs-{hyparams.epochs:.2f}'
                      f'-{dev_loss_lst[-1]:.4f}'
                      f'-{dev_acc_lst[-1]:.4f}.pkl')
    torch.save(model.state_dict(), model_filename)
    epochs = [i for i in range(hyparams.epochs)]
    export_epochs_train_dev_something_graph(epochs, train_loss_lst, dev_loss_lst, 'loss', 'Model_train_dev_loss_graph')
    export_epochs_train_dev_something_graph(epochs, train_acc_lst, dev_acc_lst, 'accuracy', 'Model_train_dev_accuracy_graph')



if __name__ == '__main__':
    main()
