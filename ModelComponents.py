import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import math


def get_rep_mask(lengths, device):
    batch_size = len(lengths)
    seq_len = torch.max(lengths)
    rep_mask = torch.FloatTensor(batch_size, seq_len).to(torch.device(device))
    rep_mask.data.fill_(1)
    for i in range(batch_size):
        rep_mask[i, lengths[i]:] = 0

    return rep_mask.unsqueeze_(-1)


# Masked softmax
def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec - max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps / (masked_sums + 1e-20)


# Directional mask
def calc_dir_mask(direction, sentence_len, device):
    mask = torch.FloatTensor(sentence_len, sentence_len).to(torch.device(device))
    mask.data.fill_(1)
    if direction == 'fw':
        mask = torch.tril(mask, diagonal=-1)
    else:
        mask = torch.triu(mask, diagonal=1)
    mask.unsqueeze_(0)
    return mask


# Representation mask for sentences of variable lengths
def calc_rep_mask(rep_mask):
    batch_size, sentence_len, _ = rep_mask.size()

    m1 = rep_mask.view(batch_size, sentence_len, 1)
    m2 = rep_mask.view(batch_size, 1, sentence_len)
    mask = torch.mul(m1, m2)

    return mask


# Implementing the calculation of the Distance mask as shown in Figure 4
def calc_dis_mask(len, device):
    mask = torch.FloatTensor(len, len).to(device)

    for i in range(len):
        for j in range(len):
            mask[i, j] = -abs(i - j)

    mask.unsqueeze_(0)

    return mask


"""
********************************************************************************
                            MODEL & MODEL COMPONENTS
********************************************************************************
"""


# Implementing the Masked Attention component mentioned in the article (4.2.2)
class MaskedAttention(nn.Module):

    def __init__(self, model_dim, dir, alpha, device='cuda:0'):
        super(MaskedAttention, self).__init__()
        self.direction = dir
        self.device = device
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=2)

        # The formula shown division by square root of d, so we'll calculate it in order to use it later
        self.scale = Variable(torch.Tensor([math.pow(model_dim, 0.5)]), requires_grad=False).cuda()

    def forward(self, q, k, v, rep_mask):
        _, seq_len, _ = q.size()

        # Implementation of the Attention calculation mentioned in formula no.6
        attention = torch.bmm(q, k.transpose(1, 2)) / self.scale

        # Implementation of the Masked calculation mentioned in formula no.7
        dir_mask = calc_dir_mask(self.direction, seq_len, self.device)
        rep_mask = calc_rep_mask(rep_mask)
        dis_mask = calc_dis_mask(seq_len, self.device)
        mask = rep_mask * dir_mask
        attention += self.alpha * dis_mask

        attention = masked_softmax(attention, mask, dim=2)
        out = torch.bmm(attention, v)

        return out, attention


# Implementing the Masked Multi-Head Attention component mentioned in the article (4.2.2)
class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, e_dim, head_count, alpha, device, dropout, dir):
        super(MaskedMultiHeadAttention, self).__init__()

        # Definition according to the article
        self.head_count = head_count
        self.k_dim = e_dim / head_count
        self.v_dim = e_dim / head_count
        self.model_dim = e_dim

        self.w_q = nn.Parameter(torch.FloatTensor(self.head_count, self.model_dim, self.k_dim))
        self.w_k = nn.Parameter(torch.FloatTensor(self.head_count, self.model_dim, self.k_dim))
        self.w_v = nn.Parameter(torch.FloatTensor(self.head_count, self.model_dim, self.v_dim))

        self.masked_attention = MaskedAttention(self.model_dim, dir, alpha, device=device)

        # Layers definition
        self.layer_norm = nn.LayerNorm(int(self.k_dim))
        self.layer_norm2 = nn.LayerNorm(self.model_dim)
        self.proj = nn.Linear(self.head_count * self.v_dim, self.model_dim)
        self.dropout = nn.Dropout(dropout)

        # Initializing weights
        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)

    def forward(self, q, k, v, rep_mask):
        head_count = self.head_count

        # Extracting initial data
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()
        mb_size, len_v, d_model = v.size()

        # Creating QW, KW, VW in order to calculate the masked attention
        q = q.repeat(head_count, 1, 1).view(head_count, -1, d_model)
        k = k.repeat(head_count, 1, 1).view(head_count, -1, d_model)
        v = v.repeat(head_count, 1, 1).view(head_count, -1, d_model)

        qw = self.layer_norm(torch.bmm(q, self.w_q).view(-1, len_q, self.k_dim))
        kw = self.layer_norm(torch.bmm(k, self.w_k).view(-1, len_k, self.k_dim))
        vw = self.layer_norm(torch.bmm(v, self.w_v).view(-1, len_v, self.v_dim))

        rep_mask = rep_mask.repeat(head_count, 1, 1).view(-1, len_q, 1)
        outs, _ = self.masked_attention(qw, kw, vw, rep_mask)

        # Implementation of the Masked Multi-Head calculation mentioned in formula no.8
        outs = torch.cat(torch.split(outs, mb_size, dim=0), dim=-1)

        # Using normalization and dropout
        outs = self.layer_norm2(self.proj(outs))
        outs = self.dropout(outs)

        return outs


# Implementing the Fusion Gate component mentioned in the article (4.2.3)
class FusionGate(nn.Module):

    def __init__(self, e_dim, dropout=0.1):
        super(FusionGate, self).__init__()

        # Creating the weights
        self.w_s = nn.Parameter(torch.FloatTensor(e_dim, e_dim))
        self.w_h = nn.Parameter(torch.FloatTensor(e_dim, e_dim))
        self.b = nn.Parameter(torch.FloatTensor(e_dim))

        # Initializing the weights
        init.xavier_normal_(self.w_s)
        init.xavier_normal_(self.w_h)
        init.constant_(self.b, 0)

        # Layers definition
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(e_dim)

    def forward(self, s, h):
        # Implementing formula no.11
        s_f = self.layer_norm(torch.matmul(s, self.w_s))
        h_f = self.layer_norm(torch.matmul(h, self.w_h))

        # Implementing formula no.12
        f = self.sigmoid(self.dropout(s_f + h_f + self.b))
        outs = f * s_f + (1 - f) * h_f

        return self.layer_norm(outs)


# Implementing the Fusion Gate component mentioned in the article (4.2.4)
class PositionwiseFeedForward(nn.Module):

    def __init__(self, e_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # Implementing the weights as defined after formula no.13
        self.w_1 = nn.Conv1d(e_dim, ff_dim, 1)
        self.w_2 = nn.Conv1d(ff_dim, e_dim, 1)

        # Layers definition
        self.layer_norm = nn.LayerNorm(e_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.w_1(x.transpose(1, 2)))
        out = self.w_2(out).transpose(2, 1)
        out = self.dropout(out)
        return self.layer_norm(out + x)


# Implementing the Pooling layer component mentioned in the article (4.2.5). They used the source2token self attention.
class Source2Token(nn.Module):

    def __init__(self, h_dim, dropout=0.1):
        super(Source2Token, self).__init__()

        self.h_dim = h_dim
        self.dropout_rate = dropout

        # Layers definition
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(h_dim)

    def forward(self, x, rep_mask):
        # Implementing formula no.14
        out = self.elu(self.layer_norm(self.fc1(x)))
        out = self.layer_norm(self.fc2(out))

        out = masked_softmax(out, rep_mask, dim=1)
        out = torch.sum(torch.mul(x, out), dim=1)

        return out


# For convenience, since each input needs to go through all of the components above, we'll batch them together in one
# class.
class LayerBatch(nn.Module):

    def __init__(self, e_dim, head_count, alpha, device, dropout, ff_dim, dir):
        super(LayerBatch, self).__init__()

        # Defining the components
        self.attention = MaskedMultiHeadAttention(e_dim, head_count, alpha, device, dropout, dir)
        self.fusion_gate = FusionGate(e_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(e_dim, ff_dim, dropout)

    def forward(self, x, rep_mask):

        # Forwarding through all of the components
        outs = self.attention(x, x, x, rep_mask)
        outs = self.fusion_gate(x, outs)
        outs = self.feed_forward(outs)

        return outs


# Implementing the Sentence Encoder component mentioned in the article (4.2)
class SentenceEncoder(nn.Module):

    def __init__(self, e_dim, head_count, alpha, device, dropout, ff_dim):
        super(SentenceEncoder, self).__init__()

        # Going through the network for each direction
        self.fw = LayerBatch(e_dim, head_count, alpha, device, dropout, ff_dim, dir='fw')
        self.bw = LayerBatch(e_dim, head_count, alpha, device, dropout, ff_dim, dir='bw')

        # Multi-dimensional source2token self-attention
        self.s2t = Source2Token(2 * e_dim, dropout)

    def forward(self, input, rep_mask):
        batch, seq_len, _ = input.size()

        # Forwarding and Backwarding the input
        forward = self.fw(input, rep_mask)
        backward = self.bw(input, rep_mask)

        # One step before the last, concat the data
        u = torch.cat([forward, backward], dim=-1)

        # Last step, going through Max Pooling
        pooling = nn.MaxPool2d((seq_len, 1), stride=1)
        pool_s = pooling(u * rep_mask).view(batch, -1)
        s2t_s = self.s2t(u, rep_mask)

        return torch.cat([s2t_s, pool_s], dim=-1)


class SNLI(nn.Module):

    def __init__(self, e_dim, head_count, alpha, device, dropout, ff_dim, out_dim, vocab_size, emb_dim, data):
        super(SNLI, self).__init__()

        # Parameters definition
        self.out_dim = out_dim
        self.dropout = dropout
        self.e_dim = e_dim
        self.ff_dim = ff_dim
        self.device = device

        # Word embedding definition
        self.emb = nn.Embedding(vocab_size, emb_dim)

        # Initializing word embedding (GloVe)
        self.emb.weight.data.copy_(data.TEXT.vocab.vectors)
        self.emb.weight.requires_grad = False

        # Random init the <unk> vectors
        nn.init.uniform_(self.emb.weight.data[0], -0.05, 0.05)

        self.sentence_encoder = SentenceEncoder(e_dim, head_count, alpha, device, dropout, ff_dim)

        # Layers definition
        self.fc = nn.Linear(e_dim * 4 * 4, e_dim)
        self.out_layer = nn.Linear(e_dim, out_dim)
        self.layer_norm = nn.LayerNorm(e_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, batch):
        premise, pre_len = batch.premise
        hypothesis, hypo_len = batch.hypothesis

        # Creating premise & hypothesis word embeddings
        pre_input = self.emb(premise)
        hypo_input = self.emb(hypothesis)

        # Getting premise & hypothesis representation masks
        pre_rep_mask = get_rep_mask(pre_len, self.device)
        hypo_rep_mask = get_rep_mask(hypo_len, self.device)

        # Passing the premise & hypothesis word embeddings through the sentence encoder
        pre_s = self.sentence_encoder(pre_input, pre_rep_mask)
        hypo_s = self.sentence_encoder(hypo_input, hypo_rep_mask)

        # Concatenating the result
        s = torch.cat([pre_s, hypo_s, (pre_s - hypo_s).abs(), pre_s * hypo_s], dim=-1)

        s = self.dropout(s)
        outputs = self.relu(self.layer_norm(self.fc(s)))
        outputs = self.dropout(outputs)
        outputs = self.out_layer(outputs)

        return outputs
