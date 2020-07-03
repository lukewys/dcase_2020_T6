import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import pickle
import torch
import numpy as np
import itertools
import inspect
import copy
from hparams import hparams as hp
from eval_metrics import evaluate_metrics
from eval_metrics import evaluate_metrics_from_lists
from eval_metrics import combine_single_and_per_file_metrics
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from gensim.models.word2vec import Word2Vec


def get_file_list(filepath, file_extension, recursive=True):
    '''
    @:param filepath: a string of directory
    @:param file_extension: a string of list of strings of the file extension wanted, format in, for example, '.xml', with the ".".
    @:return A list of all directories of files in given extension in given filepath.
    If recursive is True，search the directory recursively.
    '''
    pathlist = []
    if recursive:
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if type(file_extension) is list:
                    for exten in file_extension:
                        if file.endswith(exten):
                            pathlist.append(os.path.join(root, file))
                elif type(file_extension) is str:
                    if file.endswith(file_extension):
                        pathlist.append(os.path.join(root, file))
    else:
        files = os.listdir(filepath)
        for file in files:
            if type(file_extension) is list:
                for exten in file_extension:
                    if file.endswith(exten):
                        pathlist.append(os.path.join(filepath, file))
            elif type(file_extension) is str:
                if file.endswith(file_extension):
                    pathlist.append(os.path.join(filepath, file))
    if len(pathlist) == 0:
        print('Wrong or empty directory')
        raise FileNotFoundError
    return pathlist


def get_word_dict(word_dict_pickle_path, offset=0, reverse=False):
    word_dict_pickle = pickle.load(open(word_dict_pickle_path, 'rb'))
    word_dict = {}
    for i in range(0 + offset, len(word_dict_pickle) + offset):
        if reverse:
            word_dict[word_dict_pickle[i]] = i
        else:
            word_dict[i] = word_dict_pickle[i]
    return word_dict


def ind_to_str(sentence_ind, special_token, word_dict):
    sentence_str = []
    for s in sentence_ind:
        if word_dict[s] not in special_token:
            sentence_str.append(word_dict[s])
    return sentence_str

def gen_str(output_batch,word_dict_pickle_path):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    output_str = [' '.join(o) for o in output_str]
    return  output_str

def get_eval(output_batch, ref_batch, word_dict_pickle_path):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    ref_str = [[ind_to_str(r, special_token, word_dict) for r in ref] for ref in ref_batch]

    output_str = [' '.join(o) for o in output_str]
    ref_str = [[' '.join(r) for r in ref] for ref in ref_str]

    return  output_str, ref_str


def calculate_bleu(output, ref, word_dict_pickle_path, multi_ref=False):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = ind_to_str(output, special_token, word_dict)
    if multi_ref:
        ref_str = [ind_to_str(r, special_token, word_dict) for r in ref]
    else:
        ref_str = [ind_to_str(ref, special_token, word_dict)]

    gram_weights = []
    max_gram = 4
    for gram in range(1, max_gram + 1):
        weights = [0, 0, 0, 0]
        for i in range(gram):
            weights[i] = 1 / gram
        weights = tuple(weights)
        gram_weights.append(weights)

    score_list = []
    for weights in gram_weights:
        score = sentence_bleu(ref_str, output_str, weights=weights)
        score_list.append(score)
    return score_list, output_str, ref_str


def calculate_spider(output_batch, ref_batch, word_dict_pickle_path):
    word_dict = get_word_dict(word_dict_pickle_path)
    word_dict[len(word_dict)] = '<pad>'
    special_token = ['<sos>', '<eos>', '<pad>']
    output_str = [ind_to_str(o, special_token, word_dict) for o in output_batch]
    ref_str = [[ind_to_str(r, special_token, word_dict) for r in ref] for ref in ref_batch]

    output_str = [' '.join(o) for o in output_str]
    ref_str = [[' '.join(r) for r in ref] for ref in ref_str]

    metrics, per_file_metrics = evaluate_metrics_from_lists(output_str, ref_str)
    score = metrics['SPIDEr']

    return score, output_str, ref_str


def greedy_decode(model, src, max_len, start_symbol_ind=0):
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    # memory = model.cnn(src)
    memory = model.encode(src)
    ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    for i in range(max_len - 1):
        # ys_i:(batch_size, T_pred=i+1)
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, nhid)
        prob = model.generator(out[-1, :])  # (T_-1, batch_size, nhid)
        next_word = torch.argmax(prob, dim=1)  # (batch_size)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
        # ys_i+1: (batch_size,T_pred=i+2)
    return ys

class Beam:
    """
    The beam class for handling beam search.
    partly adapted from
    https://github.com/OpenNMT/OpenNMT-py/blob/195f5ae17f572c22ff5229e52c2dd2254ad4e3db/onmt/translate/beam.py

    There are some place which needs improvement:
    1. The prev_beam should be separated as prev_beam and beam_score.
    The prev_beam should be a tensor and beam_score should be a numpy array,
    such that the beam advance() method could speeds up.
    2. Do not support advance function like length penalty.
    3. If the beam is done searching, it could quit from further computation.
    In here, an eos is simply appended and still go through the model in next iteration.
    """

    def __init__(self, beam_size, device, start_symbol_ind, end_symbol_ind):
        self.device = device
        self.beam_size = beam_size
        self.prev_beam = [[torch.ones(1).fill_(start_symbol_ind).long().to(device), 0]]
        self.start_symbol_ind = start_symbol_ind
        self.end_symbol_ind = end_symbol_ind
        self.eos_top = False
        self.finished = []
        self.first_time = True

    def advance(self, word_probs, first_time):  # word_probs: (beam_size, ntoken) or (1, ntoken) for the first time.

        if self.done():
            # if current beam is done, just add eos to the beam.
            for b in self.prev_beam:
                b[0] = torch.cat([b[0], torch.tensor(self.end_symbol_ind).unsqueeze(0).to(self.device)])
            return

        # in first time, the beam need not to align with each index.
        if first_time:  # word_probs:(1, ntoken)
            score, index = word_probs.squeeze(0).topk(self.beam_size, 0, True, True)  # get the initial topk
            self.prev_beam = []
            for s, ind in zip(score, index):
                # initialize each beam
                self.prev_beam.append([torch.tensor([self.start_symbol_ind, ind]).long().to(self.device), s.item()])
                self.prev_beam = self.sort_beam(self.prev_beam)
        else:  # word_probs:(beam_size, ntoken)
            score, index = word_probs.topk(self.beam_size, 1, True, True)  # get topk
            current_beam = [[b[0].clone().detach(), b[1]] for b in self.prev_beam for i in range(self.beam_size)]
            # repeat each beam beam_size times for global score comparison, need to detach each tensor copied.
            i = 0
            for score_beam, index_beam in zip(score, index):  # get topk scores and corresponding index for each beam
                for s, ind in zip(score_beam, index_beam):
                    current_beam[i][0] = torch.cat([current_beam[i][0], ind.unsqueeze(0)])
                    # append current index to beam
                    current_beam[i][1] += s.item()  # add the score
                    i += 1

            current_beam = self.sort_beam(current_beam)  # sort current beam
            if current_beam[0][0][-1] == self.end_symbol_ind:  # check if the top beam ends with eos
                self.eos_top = True

            # check for eos node and added them to finished beam list.
            # In the end, delete those nodes and do not let them have child note.
            delete_beam_index = []
            for i in range(len(current_beam)):
                if current_beam[i][0][-1] == self.end_symbol_ind:
                    delete_beam_index.append(i)
            for i in sorted(delete_beam_index, reverse=True):
                self.finished.append(current_beam[i])
                del current_beam[i]

            self.prev_beam = current_beam[:self.beam_size]  # get top beam_size beam
            # print(self.prev_beam)

    def done(self):
        # check if current beam is done searching
        return self.eos_top and len(self.finished) >= 1

    def get_current_state(self):
        # get current beams
        # print(self.prev_beam)
        return torch.stack([b[0] for b in self.prev_beam])

    def get_output(self):
        if len(self.finished) > 0:
            # sort the finished beam and return the sentence with the highest score.
            self.finished = self.sort_beam(self.finished)
            return self.finished[0][0]
        else:
            self.prev_beam = self.sort_beam(self.prev_beam)
            return self.prev_beam[0][0]

    def sort_beam(self, beam):
        # sort the beam according to the score
        return sorted(beam, key=lambda x: x[1], reverse=True)


def beam_search(model, src, max_len=30, start_symbol_ind=0, end_symbol_ind=9, beam_size=1):
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    memory = model.encode(src)  # memory:(T_mem,batch_size,nhid)
    # ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    first_time = True

    beam = [Beam(beam_size, device, start_symbol_ind, end_symbol_ind) for _ in range(batch_size)]  # a batch of beams

    for i in range(max_len):
        # end if all beams are done, or exceeds max length
        if all((b.done() for b in beam)):
            break

        # get current input
        ys = torch.cat([b.get_current_state() for b in beam], dim=0).to(device).requires_grad_(False)

        # get input mask
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, ntoken) for first time,
        # (T_out, batch_size*beam_size, ntoken) in other times
        out = F.log_softmax(out[-1, :], dim=-1)  # (batch_size, ntoken) for first time,
        # (batch_size*beam_size, ntoken) in other times

        beam_batch = 1 if first_time else beam_size
        # in the first run, a slice of 1 should be taken for each beam,
        # later, a slice of [beam_size] need to be taken for each beam.
        for j, b in enumerate(beam):
            b.advance(out[j * beam_batch:(j + 1) * beam_batch, :], first_time)  # update each beam

        if first_time:
            first_time = False  # reset the flag
            # after the first run, the beam expands, so the memory needs to expands too.
            memory = memory.repeat_interleave(beam_size, dim=1)

    output = [b.get_output() for b in beam]
    return output


def get_padding(tgt, tgt_len):
    # tgt: (batch_size, max_len)
    device = tgt.device
    batch_size = tgt.size()[0]
    max_len = tgt.size()[1]
    mask = torch.zeros(tgt.size()).type_as(tgt).to(device)
    for i in range(batch_size):
        d = tgt[i]
        num_pad = max_len-int(tgt_len[i].item())
        mask[i][max_len - num_pad:] = 1
        # tgt[i][max_len - num_pad:] = pad_idx

    # mask:(batch_size,max_len)
    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()
    return mask


def print_hparams(hp):
    attributes = inspect.getmembers(hp, lambda a: not (inspect.isroutine(a)))
    return dict([a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def find_item(data, key, query, item):
    """
    Search the query in key and take out the corresponding item.
    :param data:
    :param key:
    :param query:
    :param item:
    :return:
    """
    return data[data[key] == query][item].iloc[0]


# https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
# When smoothing=0.0, the output is almost the same as nn.CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore_index:
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target.data == self.ignore_index)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def align_word_embedding(word_dict_pickle_path, w2v_model_path, ntoken, nhid):
    word_dict = get_word_dict(word_dict_pickle_path)
    model = Word2Vec.load(w2v_model_path)
    word_emb = torch.zeros((ntoken, nhid)).float()
    word_emb.uniform_(-0.1, 0.1)
    w2v_vocab = [k for k in model.wv.vocab.keys()]
    for i in range(len(word_dict)):
        word = word_dict[i]
        if word in w2v_vocab:
            w2v_vector = model.wv[word]
            word_emb[i] = torch.tensor(w2v_vector).float()
    return word_emb

if __name__ == '__main__':
    print('util')
