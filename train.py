import torch
import torch.nn as nn
import time

from data_handling import get_clotho_loader,get_test_data_loader
from model import TransformerModel  # , RNNModel, RNNModelSmall
import itertools
import numpy as np
import os
import sys
import logging
import csv

from util import get_file_list,  get_padding, print_hparams, greedy_decode, \
    calculate_bleu, calculate_spider, LabelSmoothingLoss,beam_search,align_word_embedding,gen_str
from hparams import hparams
from torch.utils.tensorboard import SummaryWriter

import argparse

hp = hparams()
parser = argparse.ArgumentParser(description='hparams for model')

device = torch.device(hp.device)
np.random.seed(hp.seed)
torch.manual_seed(hp.seed)


def train():
    model.train()
    total_loss_text = 0.
    start_time = time.time()
    batch = 0
    for src, tgt,tgt_len, ref in training_data:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_pad_mask = get_padding(tgt, tgt_len)
        tgt_in = tgt[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_in, target_padding_mask=tgt_pad_mask)

        loss_text = criterion(output.contiguous().view(-1, hp.ntoken), tgt_y.transpose(0, 1).contiguous().view(-1))
        loss = loss_text
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad)
        optimizer.step()
        total_loss_text += loss_text.item()

        writer.add_scalar('Loss/train-text', loss_text.item(), (epoch - 1) * len(training_data) + batch)

        batch += 1

        if batch % hp.log_interval == 0 and batch > 0:
            mean_text_loss = total_loss_text / hp.log_interval
            elapsed = time.time() - start_time
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} | ms/batch {:5.2f} | '
                         'loss-text {:5.4f}'.format(
                epoch, batch, len(training_data), current_lr,
                elapsed * 1000 / hp.log_interval, mean_text_loss))
            total_loss_text = 0
            start_time = time.time()


def eval_all(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None):
    model.eval()
    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt,_, ref in evaluation_data:
            src = src.to(device)
            output = greedy_decode(model, src, max_len=max_len)

            output_sentence_ind_batch = []
            for i in range(output.size()[0]):
                output_sentence_ind = []
                for j in range(1, output.size(1)):
                    sym = output[i, j]
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym.item())
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)
        score, output_str, ref_str = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        writer.add_scalar(f'Loss/eval_greddy', loss_mean, epoch)
        msg = f'eval_greddy SPIDEr: {loss_mean:2.4f}'
        logging.info(msg)

def eval_with_beam(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None,beam_size = 3):
    model.eval()
    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt,_, ref in evaluation_data:
            src = src.to(device)
            output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)

            output_sentence_ind_batch = []
            for single_sample in output:
                output_sentence_ind = []
                for sym in single_sample:
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym.item())
                output_sentence_ind_batch.append(output_sentence_ind)
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)

        score, output_str, ref_str = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        writer.add_scalar(f'Loss/eval_beam', loss_mean, epoch)
        msg = f'eval_beam_{beam_size} SPIDEr: {loss_mean:2.4f}'
        logging.info(msg)

def test_with_beam(test_data, max_len=30, eos_ind=9, beam_size=3):
    model.eval()

    with torch.no_grad():
        with open("test_out.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'caption_predicted'])
            for src, filename in test_data:
                src = src.to(device)
                output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)

                output_sentence_ind_batch = []
                for single_sample in output:
                    output_sentence_ind = []
                    for sym in single_sample:
                        if sym == eos_ind: break
                        output_sentence_ind.append(sym.item())
                    output_sentence_ind_batch.append(output_sentence_ind)
                out_str  = gen_str(output_sentence_ind_batch,hp.word_dict_pickle_path)
                for caption, fn in zip(out_str, filename):
                    writer.writerow(['{}.wav'.format(fn),caption])


if __name__ == '__main__':
    parser.add_argument('--device', type=str, default=hp.device)
    parser.add_argument('--nlayers', type=int, default=hp.nlayers)
    parser.add_argument('--nhead', type=int, default=hp.nhead)
    parser.add_argument('--nhid', type=int, default=hp.nhid)
    parser.add_argument('--training_epochs', type=int, default=hp.training_epochs)
    parser.add_argument('--lr', type=float, default=hp.lr)
    parser.add_argument('--scheduler_decay', type=float, default=hp.scheduler_decay)
    parser.add_argument('--load_pretrain_cnn',action='store_true')
    parser.add_argument('--load_pretrain_emb', action='store_true')
    parser.add_argument('--load_pretrain_model', action='store_true')
    parser.add_argument('--spec_augmentation',action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--name', type=str, default=hp.name)
    parser.add_argument('--pretrain_emb_path',type=str, default=hp.pretrain_emb_path)
    parser.add_argument('--pretrain_cnn_path', type=str, default=hp.pretrain_cnn_path)
    parser.add_argument('--pretrain_model_path', type=str, default=hp.pretrain_model_path)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(hp, k, v)
    args = parser.parse_args()

    pretrain_emb = align_word_embedding(hp.word_dict_pickle_path, hp.pretrain_emb_path, hp.ntoken,
                                        hp.nhid) if hp.load_pretrain_emb else None
    pretrain_cnn = torch.load(hp.pretrain_cnn_path) if hp.load_pretrain_cnn else None

    model = TransformerModel(hp.ntoken, hp.ninp, hp.nhead, hp.nhid, hp.nlayers, hp.batch_size, dropout=0.2,
                            pretrain_cnn=pretrain_cnn,pretrain_emb=pretrain_emb).to(device)
    if hp.load_pretrain_model:
        model.load_state_dict(torch.load(hp.pretrain_model_path))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, hp.scheduler_decay)
    if hp.label_smoothing:
        criterion = LabelSmoothingLoss(hp.ntoken, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=hp.ntoken - 1)

    now_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
    log_dir = 'models/{name}'.format(name = hp.name)

    writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, 'train.log')

    logging.basicConfig(level=logging.DEBUG,
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)]
                        )

    data_dir = hp.data_dir
    eval_data_dir = hp.eval_data_dir
    train_data_dir = hp.train_data_dir
    word_dict_pickle_path = hp.word_dict_pickle_path
    word_freq_pickle_path = hp.word_freq_pickle_path
    test_data_dir = hp.test_data_dir

    training_data = get_clotho_loader(data_dir=data_dir, split='development',
                                      input_field_name='features',
                                      output_field_name='words_ind',
                                      load_into_memory=False,
                                      batch_size=hp.batch_size,
                                      nb_t_steps_pad='max',
                                      num_workers=4, return_reference=True, augment =  hp.spec_augmentation)

    evaluation_beam = get_clotho_loader(data_dir=data_dir, split='evaluation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=32,
                                        nb_t_steps_pad='max',
                                        shuffle=False,
                                        return_reference=True)
    test_data = get_test_data_loader(data_dir=test_data_dir,
                                     batch_size=hp.batch_size * 2,
                                     nb_t_steps_pad='max',
                                     shuffle=False,
                                     drop_last=False,
                                     input_pad_at='start',
                                     num_workers=8)
    logging.info(str(model))

    logging.info(str(print_hparams(hp)))

    logging.info('Data loaded!')
    logging.info('Data size: ' + str(len(evaluation_beam )))

    logging.info('Total Model parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    epoch = 1
    if hp.mode == 'train':
        while epoch < hp.training_epochs + 1:
            epoch_start_time = time.time()
            train()
            torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            scheduler.step(epoch)
            eval_all(evaluation_beam, word_dict_pickle_path=word_dict_pickle_path)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=2)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=3)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=4)
            epoch += 1

    if hp.mode == 'eval':
        #Evaluation model score
        model.load_state_dict(torch.load("./models/best.pt"))
        eval_all(evaluation_beam, word_dict_pickle_path=word_dict_pickle_path)
        eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                       beam_size=2)
        eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                       beam_size=3)
        eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                       beam_size=4)

    elif hp.mode == 'test':
        #Generate caption(in test_out.csv)
        model.load_state_dict(torch.load("./models/best.pt"))
        test_with_beam(test_data, beam_size=3)
