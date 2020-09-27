import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import prune_scheduler

import torchtext
from torchtext.data.utils import get_tokenizer
from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter

TEXT = torchtext.data.Field(
    tokenize=get_tokenizer("basic_english"),
    init_token='<sos>',
    eos_token='<eos>',
    lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, ntoken=None, emsize=None, nhid=None, nlayers=None, nhead=None, dropout=None):
        super(Transformer, self).__init__()

        # Move this elsewhere (to dataloader)
        ntoken = len(TEXT.vocab.stoi)

        #emsize == n_inputs
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, emsize)
        self.ninp = emsize
        self.decoder = nn.Linear(emsize, ntoken)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        #from IPython import embed; embed()
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class TransformerLearner():
    def __init__(self, model_config=None, num_epochs=5, target_sparsity=0.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self._init_model(model_config)
        self.crit = nn.CrossEntropyLoss()

        self.model.to(self.device)
        self.crit.to(self.device)

        self.optim = torch.optim.SGD(self.model.parameters(), lr=5.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 1.0, gamma=0.95)

        self.writer = SummaryWriter()

        # Array of sparsity rates
        self.sparsity_scheduler = prune_scheduler(target_sparsity, num_epochs, start_epoch=0)

    def _init_model(self, model_config):
        return Transformer(**model_config)

    def train(self, epoch):
        self.model.train()

        total_loss = 0.
        start_time = time.time()
        ntokens = len(TEXT.vocab.stoi)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            self.optim.zero_grad()
            data, targets = get_batch(train_data, i)

            output = self.model(data)
            loss = self.crit(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optim.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, len(train_data) // bptt, self.scheduler.get_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

        self.scheduler.step()

    def eval(self, epoch):
        data_source = val_data
        self.model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(TEXT.vocab.stoi)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                data = data.cuda()
                targets = targets.cuda()
                output = self.model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * self.crit(output_flat, targets).item()

        val_loss = total_loss / (len(data_source) - 1)
        print("-" * 89)
        print('| end of epoch {:3d} | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, val_loss, math.exp(val_loss)))
        print('-' * 89)

    def prune_weights(self, epoch):
        # Send weights to cpu
        sparsity_rate = self.sparsity_scheduler[epoch]
        print(sparsity_rate)

        self.modules_to_prune = []
        for name, module in self.model.named_modules():
            if type(module) == torch.nn.modules.linear.Linear:
                self.modules_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            self.modules_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity_rate)

        self._report_sparsity(epoch)

    def _report_sparsity(self, epoch, verbose=False):
        pruned_modules = []
        for name, module in self.model.named_modules():
            if type(module) == torch.nn.modules.linear.Linear:
                nz = float(torch.sum(module.weight == 0))
                nelement = float(module.weight.nelement())
                pruned_modules.append((name, nz, nelement))

        t_nz = 0
        t_nelement = 0

        # Report
        for name, nz, nelement in pruned_modules:
            t_nz += nz
            t_nelement += nelement
            if verbose:
                print("Sparsity in {}: {:2f}%".format(name, 100. * nz / nelement))

        print("Global sparsity: {:2f}, t_nz: {}, t_el: {}".format(100. * t_nz / t_nelement, int(t_nz), int(t_nelement)))
        self.writer.add_scalar("Global sparsity", t_nz / t_nelement, epoch)
