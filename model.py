import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerDecoder,TransformerDecoderLayer

from hparams import hparams as hp
from encoder import Cnn10,init_layer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, batch_size, dropout=0.5,pretrain_cnn=None,
                 pretrain_emb=None,freeze_cnn=True):
        super(TransformerModel, self).__init__()

        self.model_type = 'cnn+transformer'
        decoder_layers = TransformerDecoderLayer(d_model=nhid, nhead=nhead, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.word_emb = nn.Embedding(ntoken, nhid)
        self.ninp = ninp
        self.nhid = nhid
        self.fc = nn.Linear(512, 512, bias=True)
        self.fc1 = nn.Linear(512, nhid, bias=True)
        self.dec_fc = nn.Linear(nhid, ntoken)
        self.batch_size = batch_size
        self.ntoken = ntoken
        self.encoder = Cnn10()
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.generator = nn.Softmax(dim=-1)
        self.init_weights()

        if pretrain_cnn is not None:
            dict_trained = pretrain_cnn
            dict_new = self.encoder.state_dict().copy()
            new_list = list(self.encoder.state_dict().keys())
            trained_list = list(dict_trained.keys())
            for i in range(len(new_list)):
                dict_new[new_list[i]] = dict_trained[trained_list[i]]
            self.encoder.load_state_dict(dict_new)
        if freeze_cnn:
            self.freeze_cnn()

        if pretrain_emb is not None:
            self.word_emb.weight.data = pretrain_emb

    def freeze_cnn(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        init_layer(self.fc1)
        init_layer(self.fc)
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.dec_fc.bias.data.zero_()
        self.dec_fc.weight.data.uniform_(-initrange, initrange)

    def encode(self, src, input_mask=None):
        x = self.encoder(src)  # (batch_size, 512, T/16, mel_bins/16)
        x = torch.mean(x, dim=3)  # (batch_size, 512, T/16)
        x = x.permute(2, 0, 1)  # (T/16,batch_size,512)
        x = F.relu_(self.fc(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.relu(self.fc1(x))
        return x

    def decode(self, mem, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt:(batch_size,T_out)
        # mem:(T_mem,batch_size,nhid)

        tgt = tgt.transpose(0, 1)  # (T_out,batch_size)
        if target_mask is None or target_mask.size(0) != len(tgt):
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.dropout(self.word_emb(tgt)) * math.sqrt(self.nhid)
        tgt = self.pos_encoder(tgt)
        # mem = self.pos_encoder(mem)
        output = self.transformer_decoder(tgt, mem, memory_mask=input_mask, tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)
        return output

    def forward(self, src, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # src:(batch_size,T_in,feature_dim)
        # tgt:(batch_size,T_out)
        mem = self.encode(src)
        output = self.decode(mem, tgt, input_mask=input_mask, target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        return output
