import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from pathlib import Path
from data_handling import get_clotho_loader,get_test_data_loader
from model import TransformerModel  # , RNNModel, RNNModelSmall
# from model3 import Cnn10
# from model4 import AudioCaptionEncoderDecoder, TransformerCaptionDecoder, TaggingClassifier
import itertools
import numpy as np
import os
import sys
import logging
import csv

from util import get_file_list, eval_and_bleu, get_padding, print_hparams, get_tag_batch_from_ref, greedy_decode, \
    calculate_bleu, calculate_spider, LabelSmoothingLoss,beam_search,align_word_embedding,gen_str
from hparams import hparams as hp
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import argparse

parser = argparse.ArgumentParser(description='hparams for model')

device = torch.device("cuda:1" if hp.cuda else "cpu")
np.random.seed(hp.seed)
torch.manual_seed(hp.seed)


def train():
    # TODO: warmup, schedule sampling, beamsearch
    # TODO: 一定概率的teacher forcing, 尝试conv1d+avgpool：https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556
    # TODO: 加事件加测：应该删除率高一些
    # TODO: 加入baseline中的evaluation方法
    model.train()
    total_loss_text = 0.
    total_loss_tagging = 0.
    start_time = time.time()
    batch = 0
    for src, tgt,tgt_len, ref, tag_target in training_data:

        tag_target = tag_target.to(device)

        src = src.to(device)
        tgt = tgt.to(device)
        tgt_pad_mask = get_padding(tgt, tgt_len)
        tgt_in = tgt[:, :-1]
        tgt_pad_mask = tgt_pad_mask[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()
        output, output_tag = model(src, tgt_in, target_padding_mask=tgt_pad_mask)

        # model.eval()
        # greedy_decode(model, src[0].unsqueeze(0), 30)

        loss_tag = tag_loss(output_tag, tag_target) if hp.tagging else torch.zeros(1)

        loss_text = criterion(output.contiguous().view(-1, hp.ntoken), tgt_y.transpose(0, 1).contiguous().view(-1))

        if hp.tagging:
            loss = loss_text + loss_tag
        else:
            loss = loss_text
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad)
        optimizer.step()
        total_loss_text += loss_text.item()
        total_loss_tagging += loss_tag.item()

        writer.add_scalar('Loss/train-text', loss_text.item(), (epoch - 1) * len(training_data) + batch)
        writer.add_scalar('Loss/train-tag', loss_tag.item(), (epoch - 1) * len(training_data) + batch)

        batch += 1

        if batch % hp.log_interval == 0 and batch > 0:
            if epoch == 10:
                print('pause')
            mean_text_loss = total_loss_text / hp.log_interval
            mean_tagging_loss = total_loss_tagging / hp.log_interval
            elapsed = time.time() - start_time
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2e} | ms/batch {:5.2f} | '
                         'loss-text {:5.4f} | loss-tagging {:02.3e}'.format(
                epoch, batch, len(training_data), current_lr,
                elapsed * 1000 / hp.log_interval, mean_text_loss, mean_tagging_loss))
            total_loss_text = 0
            total_loss_tagging = 0
            start_time = time.time()


def eval(evaluation_data_dir, num_eval=10, print_prob=1, max_len=30, eos_ind=9, word_dict_pickle_path=None, flag='',
         bleu_gram_max=4):
    model.eval()
    eval_data_list = get_file_list(evaluation_data_dir, '.npy')
    total_loss = np.zeros(bleu_gram_max)
    with torch.no_grad():
        for i in range(num_eval):
            data_ind = np.random.randint(0, len(eval_data_list) - 1)
            score, filename, output_str, ref_str = eval_and_bleu(eval_data_list, data_ind, model, word_dict_pickle_path,
                                                                 max_len=max_len, eos_ind=eos_ind, device=device,
                                                                 multi_ref=True)

            msg = f'Filename-{flag}:' + filename + '\n' + f'Output-{flag}:' + str(
                output_str) + '\n' + f'Ref-{flag}:' + str(ref_str) + '\n' + 'Score: ' + str(score)
            writer.add_text(f'Text/{flag}-bleu', msg, epoch)

            if np.random.rand() < print_prob:
                logging.info(msg)

            total_loss += np.array(score)
        loss_mean = total_loss / num_eval
        for i in range(bleu_gram_max):
            writer.add_scalar(f'Loss/{flag}_bleu{i + 1}', loss_mean[i], epoch)

        msg = f'{flag} bleu: '
        for i in range(bleu_gram_max):
            msg = msg + f'{loss_mean[i]:5.4f} | '
        logging.info(msg)


def eval_all(evaluation_data, max_len=30, eos_ind=9, word_dict_pickle_path=None):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        output_sentence_all = []
        ref_all = []
        for src, tgt,_, ref, tag_target in evaluation_data:
            src = src.to(device)
            output = greedy_decode(model, src, max_len=max_len)

            output_sentence_ind_batch = []
            for i in range(output.size()[0]):
                output_sentence_ind = []
                for j in range(1, output.size(1)):
                    sym = output[i, j]
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym.item())
                output_sentence_ind_batch.append(output_sentence_ind)  # TODO: 这个和单个的greedy_decode合起来写成函数
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
        for src, tgt,_, ref, tag_target in evaluation_data:
            src = src.to(device)
            output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size)

            output_sentence_ind_batch = []
            for single_sample in output:
                output_sentence_ind = []
                for sym in single_sample:
                    if sym == eos_ind: break
                    output_sentence_ind.append(sym.item())
                output_sentence_ind_batch.append(output_sentence_ind)  # TODO: 这个和单个的greedy_decode合起来写成函数
            output_sentence_all.extend(output_sentence_ind_batch)
            ref_all.extend(ref)

        score, output_str, ref_str = calculate_spider(output_sentence_all, ref_all, word_dict_pickle_path)

        loss_mean = score
        writer.add_scalar(f'Loss/eval_beam', loss_mean, epoch)
        msg = f'eval_beam_{beam_size} SPIDEr: {loss_mean:2.4f}'
        logging.info(msg)

def test_with_beam(model,test_data, max_len=30, eos_ind=9, beam_size=3):
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
    # parser.add_argument('--nlayers', type=int, default=2)
    # parser.add_argument('--nhead', type=int, default=4)
    # parser.add_argument('--nhid', type=int, default=256)
    # parser.add_argument('--training_epochs', type=int, default=1000)
    # parser.add_argument('--relative_attention', action='store_true')
    # parser.add_argument('--tagging', action='store_true')
    # args = parser.parse_args()
    # for k, v in vars(args).items():
    #     setattr(hp, k, v)
    load_pretrain_emb = False
    hpretrain_emb_path = r'w2v_192.mod'
    pretrain_emb = align_word_embedding(hp.word_dict_pickle_path, hpretrain_emb_path, hp.ntoken,
                                        hp.nhid) if load_pretrain_emb else None
    lr = 3e-4
    # lr = 1e-5
    # encoder = Cnn10().to(device)
    # decoder = TransformerCaptionDecoder(hp.ntoken, hp.nhead, hp.nhid, hp.nlayers, dropout=0.2,
    #                                     pretrain_emb=pretrain_emb, emb_bind=False).to(device)
    # tagging = TaggingClassifier(hp.nhid, hp.ntoken) if hp.tagging else None
    # model = AudioCaptionEncoderDecoder(encoder=encoder, decoder=decoder, tagging=tagging).to(device)

    model = TransformerModel(hp.ntoken, hp.ninp, hp.nhead, hp.nhid, hp.nlayers, hp.batch_size, dropout=0.2,
                             relative_transformer=hp.relative_attention, tagging=hp.tagging, pretrain_cnn=False,
                             pretrain_emb=pretrain_emb, cnntrans=False).to(device)
    # model.load_state_dict(torch.load("./logs/logs_transformer/6.16-2/8.pt"))
    # model.load_state_dict(torch.load("./logs/logs_transformer/emb2-enc/8.pt"))
    # model = torch.load("./models/best.pt").to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=1e-6)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
    if hp.label_smoothing:
        criterion = LabelSmoothingLoss(hp.ntoken, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=hp.ntoken - 1)

    tag_loss = nn.BCEWithLogitsLoss()

    now_time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))
    log_dir = 'logs/logs_{model}/{time}'.format(time=now_time, model=model.model_type)

    if hp.restore_path:
        model = torch.load(hp.restore_path)
        log_dir = os.path.dirname(hp.restore_path)

    writer = SummaryWriter(log_dir=log_dir)

    log_path = os.path.join(log_dir, 'train.log')

    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s',
                        # 日志格式
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
                                      num_workers=4, return_reference=True, augment = True)
    evaluation_data = get_clotho_loader(data_dir=data_dir, split='evaluation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=hp.batch_size,
                                        nb_t_steps_pad='max',
                                        num_workers=4, return_reference=True)  # 这个已经改成了src,tgt,ref了

    evaluation_beam = get_clotho_loader(data_dir=data_dir, split='evaluation',
                                        input_field_name='features',
                                        output_field_name='words_ind',
                                        load_into_memory=False,
                                        batch_size=32,
                                        nb_t_steps_pad='max',
                                        shuffle=False,
                                        return_reference=True)  # 这个已经改成了src,tgt,ref了
    test_data = get_test_data_loader(data_dir=test_data_dir,
                                     batch_size=hp.batch_size * 2,
                                     nb_t_steps_pad='max',
                                     shuffle=False,
                                     drop_last=False,
                                     input_pad_at='start',
                                     num_workers=8)
    # training_data_fixed = next(iter(training_data))

    logging.info(str(model))

    logging.info(str(print_hparams(hp)))

    logging.info('Data loaded!')
    logging.info('Data size: ' + str(len(evaluation_beam )))

    logging.info('Total Model parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    epoch = int(os.path.basename(hp.restore_path).replace('.pt', '')) if hp.restore_path else 1

    if hp.mode == 'train':
        while epoch < hp.training_epochs + 1:
            epoch_start_time = time.time()
            train()
            torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            scheduler.step(epoch)
            # eval(eval_data_dir, num_eval=3, word_dict_pickle_path=word_dict_pickle_path, flag='eval')
            # eval(train_data_dir, num_eval=1, word_dict_pickle_path=word_dict_pickle_path, flag='training')
            eval_all(evaluation_beam, word_dict_pickle_path=word_dict_pickle_path)
            # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,beam_size=1)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=2)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=3)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=4)
            # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
            #                            beam_size=5)

            # torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            epoch += 1

    elif hp.mode == 'eval':
        while epoch < hp.training_epochs + 1:
            epoch_start_time = time.time()
            if epoch<=10:
                model.load_state_dict(torch.load("./logs/logs_transformer/cnn_trans-10/{}.pt".format(str(epoch))))
            else:
                model.load_state_dict(torch.load("./logs/logs_transformer/cnn_trans256/{}.pt".format(str(epoch-10))))
            # train()
            # scheduler.step(epoch)
            # eval(eval_data_dir, num_eval=3, word_dict_pickle_path=word_dict_pickle_path, flag='eval')
            # eval(train_data_dir, num_eval=1, word_dict_pickle_path=word_dict_pickle_path, flag='training')
            # eval_all(evaluation_data, word_dict_pickle_path=word_dict_pickle_path)
            logging.info('epoch:'+str(epoch))
            # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,beam_size=1)
            eval_all(evaluation_data, word_dict_pickle_path=word_dict_pickle_path)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=2)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=3)
            eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
                           beam_size=4)
            # eval_with_beam(evaluation_beam, max_len=30, eos_ind=9, word_dict_pickle_path=word_dict_pickle_path,
            #                            beam_size=5)

            torch.save(model.state_dict(), '{log_dir}/{num_epoch}.pt'.format(log_dir=log_dir, num_epoch=epoch))
            epoch += 1
    elif hp.mode == 'test':
        test_with_beam(test_data, beam_size=3,)
