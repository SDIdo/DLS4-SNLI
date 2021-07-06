from sys import argv
import torch
from torch.nn.utils.rnn import pad_sequence
import json
from train_model import NLIModel
from torch.utils.data import DataLoader

STUDENTS={'name1': 'Ido Natan',
         'ID1': '305727802',
         'name2': 'Tzach Cohen',
         'ID2': '208930842'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORDS = 400002
DIMENSIONS = 50
UNKNOWN = "UNKNOWN"

class TestArgs():
    def __init__(self, model_path, test_path='data/snli_1.0_test.jsonl', indexed_words_path='indexed_words.json'):
        self.test_path = test_path
        self.model_path = model_path
        self.indexed_words_path = indexed_words_path

def create_random_embedding(words, dimension):
    return torch.rand(words, dimension)

def pad_collate(batch):
  (x1, x2, l) = zip(*batch)
  x1_lens = [len(p) for p in x1]
  x2_lens = [len(h) for h in x2]

  x1_pad = pad_sequence(x1, batch_first=True, padding_value=0)
  x2_pad = pad_sequence(x2, batch_first=True, padding_value=0)

  return x1_pad, x2_pad, x1_lens, x2_lens, l

def create_test_set(premises, labels, hypotheses, indexed_words, indexed_labels):
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

        premises_tensors.append(torch.cat([torch.tensor([indexed_words.get(w)], dtype=torch.long) for w in premise]))
        hypotheses_tensors.append(torch.cat([torch.tensor([indexed_words.get(w)], dtype=torch.long) for w in hypothesis]))
        labels_tensors.append(torch.tensor(indexed_labels[label]))

    for p, h, l in zip(premises_tensors, hypotheses_tensors, labels_tensors):
        dataset.append((p, h, l))

    print("Creation complete!")
    return dataset

def parse_test_data(file):
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

def batched_key_value(dict, keys):
    values = []
    for key in keys:
        values.append(dict[key.item()])
    return values

def pred_routin(args):
    # lstm_hidden_dims = [int(d) for d in args.lstm_hidden_dims.split(',')]
    print('Building model...')
    vecs = create_random_embedding(NUM_WORDS, DIMENSIONS) # TODO ok to say 400k words?
    model = NLIModel(vecs=vecs) # TODO trying to save the cost of loading actual vecs from json.. hoping in the load_state
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print('Testing model...')
    indexed_labels = {'-': -1, "entailment": 0, "neutral": 1, "contradiction": 2}
    indexed_labels_rev = {-1: 'not sure', 0: "entailment", 1: "neutral", 2: "contradiction"}
    indexed_words_file = open(args.indexed_words_path, 'r')
    indexed_words = json.load(indexed_words_file)
    test_premises, test_labels, test_hypotheses = parse_test_data(args.test_path)
    test_dataset = create_test_set(test_premises, test_labels, test_hypotheses, indexed_words, indexed_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    preds = []
    num_correct = 0
    num_data = len(test_dataset)
    for batch in test_dataloader:
        premise, hypothesis, premise_length, hypothesis_length, label = batch
        premise = premise.to(DEVICE)
        hypothesis = hypothesis.to(DEVICE)
        label = torch.tensor(label).to(DEVICE)
        model_output = model(premise, premise_length, hypothesis, hypothesis_length)
        label_pred = model_output.max(1)[1]
        preds.append(batched_key_value(indexed_labels_rev, label_pred)) # TODO if time add the actual prem and hypo considering the batch..
        num_correct_batch = torch.eq(label, label_pred).long().sum()
        num_correct_batch = num_correct_batch.item()
        num_correct += num_correct_batch
    with open('test.pred', "w") as f:
        f.write('Assess model on the snli testset\n')
        f.write('Students: Ido Nathan 30572782 | Tzach Cohen 208930842\n')
        f.write('************************************\n')
        for pred in preds:
            for label in pred:
                f.write(str(label))
                f.write("\n")
        f.write("\n")
        f.write(f'Model Accuracy: {round(num_correct / num_data,3)}')

if __name__ == "__main__":
    args = TestArgs(*argv[1:])
    pred_routin(args)
