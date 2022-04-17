# -*- coding: utf-8 -*-

import sys
import os
import argparse
import time
import datetime
from Time_count import time_consum
import logging
import numpy as np
import json
import random

import torch
import torch.nn as nn
from torch.backends import cudnn

# from model import Encoder, Decoder
from model_CGM_T import Encoder_rnn_vid, Decoder_command, Encoder_vid_embed, Encoder_ch
from utils import set_logger, read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, clip_gradient, \
    adjust_learning_rate, build_vocab_count
from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from cocoeval_iit_v2c import val_iit_v2c, METRICS

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

import torch.nn.functional as F

cc = SmoothingFunction()


# 这个文件是整个工程的流程，各个细节函数需要从其他py中调用。
# 大致需要：读参数--建词表--读数据--过模型--训练验证测试--存各种数据和文件。

startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))  # 把传入的元组按照格式，输出字符串
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from logs_txt.log_save import Logger
# from logs_txt.log_test import p_test
sys.stdout = Logger('./logs_txt/logs.txt', sys.stdout)  # 控制台输出日志
sys.stderr = Logger('./logs_txt/logs.txt', sys.stderr)  # 错误输出日志
print('-------------------------------------------------------', file=sys.stdout, flush=True)

# Training settings
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True  # 不过实际上这个设置对精度影响不大，
    # 仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。

seed_everything(2021)


class Arguments():  # 读参数，都是从configs.yaml读出来的
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])


def save_checkpoint(state, cp_file):  # 存cp
    torch.save(state, cp_file)


def count_paras(encoder, decoder, logging=None):
    '''
    计算模型的总参数，跑起来大概是encoder 12m，decoder 23m，一共35m的参数量
    '''
    nparas_enc = sum(p.numel() for p in encoder.parameters())  # numel函数取p的个数
    nparas_dec = sum(p.numel() for p in decoder.parameters())
    nparas_sum = nparas_enc + nparas_dec
    if logging is None:  # 打印结果
        print('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc / 1e6, nparas_dec / 1e6,
                                                                      nparas_sum / 1e6))
    else:
        logging.info('#paras of my model: enc {}M  dec {}M total {}M'.format(nparas_enc / 1e6, nparas_dec / 1e6,
                                                                             nparas_sum / 1e6))


def setup(args, clear=False):
    '''
    主要就是构建词表vocabs.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH  # 中文词表和英文词表的路径
    if clear:  ## 删除已经有的词表
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # 构建English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'), TRAIN_VOCAB_EN)
    # 构建Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # # 设定随机种子
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

# total_train_loss.append(train_loss)
# total_val_loss.append(val_loss)
def plot_epoch_loss(total_train_loss, total_val_loss, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 15))

    x = range(1, len(total_train_loss)+1)
    plt.plot(x, total_train_loss, '.-', color='blue', label='train')
    plt.plot(x, total_val_loss, '.-', color='red', label='eval')
    plt_title = 'train_eval_loss'
    plt.title(plt_title)
    plt.xlabel('epoch_num')
    plt.ylabel(plt_title)
    plt.legend()
    plt.grid()
    # plt.savefig(plt_title + '{}.jpg'.format(n))
    plt.savefig(save_path + plt_title + '.jpg')
    # plt.show()

def plot_epoch_eval_result_total(eval_result_total, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 15))

    x = range(1, len(eval_result_total)+1)
    plt.plot(x, eval_result_total, '.-', color='darkorchid', label='eval_result_total')
    plt_title = 'eval_result_total'
    plt.title(plt_title)
    plt.xlabel('epoch_num')
    plt.ylabel(plt_title)
    plt.legend()
    plt.grid()
    # plt.savefig(plt_title + '{}.jpg'.format(n))
    plt.savefig(save_path + plt_title + '.jpg')
    # plt.show()

def plot_epoch_bleu4(eval_result_total, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 15))

    x = range(1, len(eval_result_total)+1)
    plt.plot(x, eval_result_total, '.-', color='darkorchid', label='epoch_bleu4')
    plt_title = 'plot_epoch_bleu4'
    plt.title(plt_title)
    plt.xlabel('epoch_num')
    plt.ylabel(plt_title)
    plt.legend()
    plt.grid()
    # plt.savefig(plt_title + '{}.jpg'.format(n))
    plt.savefig(save_path + plt_title + '.jpg')
    # plt.show()

def plot_epoch_result_total_bleu4(scorer_sum_list, scorer_bleu4_list, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 15))

    x = range(1, len(scorer_sum_list)+1)
    plt.plot(x, scorer_sum_list, '.-', color='darkorchid', label='eval_result_total')
    plt.plot(x, scorer_bleu4_list, '.-', color='darkcyan', label='epoch_bleu4')
    plt_title = 'epoch_result_total_bleu4'
    plt.title(plt_title)
    plt.xlabel('epoch_num')
    plt.ylabel(plt_title)
    plt.legend()
    plt.grid()
    # plt.savefig(plt_title + '{}.jpg'.format(n))
    plt.savefig(save_path + plt_title + '.jpg')
    # plt.show()

# action_acc_train.append(action_test_acc_train)
# action_acc_val.append(action_test_acc_val)
def plot_epoch_action_acc(action_acc_train, action_acc_val, save_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 15))

    x = range(1, len(action_acc_train)+1)
    plt.plot(x, action_acc_train, '.-', color='blue', label='train')
    plt.plot(x, action_acc_val, '.-', color='red', label='eval')
    plt_title = 'train_eval_action_acc'
    plt.title(plt_title)
    plt.xlabel('epoch_num')
    plt.ylabel(plt_title)
    plt.legend()
    plt.grid()
    # plt.savefig(plt_title + '{}.jpg'.format(n))
    plt.savefig(save_path + plt_title + '.jpg')
    # plt.show()

def main(args):
    # model_prefix = '{}_{}'.format('CGM', 'T_NO_MASK')
    model_prefix = '{}_{}'.format('CGM', 'T')

    # ----------------------------------------------------
    # CGM_T_NO_MASK
    # len(sentence_del_n) >= 300
    #


    # 各个路径的参数
    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth"
    # cp_file = checkpoint_path + "best_bleu4.pth"
    init_epoch = 0

    # 创建对应的文件夹
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## 初始化log
    set_logger(os.path.join(log_path, 'train.log'))

    ## 保存参数，即copy一份cogfigs.yaml方便复现
    with open(log_path + 'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix))  # 写log

# ----------------------------------------------------------------------------------------------------------

    ## 构建相应的词表
    setup(args, clear=True)
    print(args.__dict__)

    word_bias_vector_en, word_weight_en, action_weight = build_vocab_count(args.DATA_DIR, language='en')
    word_bias_vector_zh, word_weight_zh, action_weight = build_vocab_count(args.DATA_DIR, language='zh')

    # import torch
    word_weight_en = torch.FloatTensor(word_weight_en)
    word_weight_en[word_weight_en > 20.0] = 20.0
    word_weight_en[word_weight_en < 0.5] = 0.5

    word_weight_zh = torch.FloatTensor(word_weight_zh)
    word_weight_zh[word_weight_zh > 35.0] = 35.0
    word_weight_zh[word_weight_zh < 0.5] = 0.5

    action_weight = torch.FloatTensor(action_weight)
    action_weight[action_weight > 10.0] = 10.0
    action_weight[action_weight < 0.5] = 0.5

    # 设置src and tgt的语言，模型可以同时做中翻英或者英翻中，需要在此处设置
    # src, tgt = 'en', 'zh'
    src, tgt = 'zh', 'en'

    maps = {'en': args.TRAIN_VOCAB_EN, 'zh': args.TRAIN_VOCAB_ZH}  # 这个maps字典存的是对应词表地址

    vocab_src = read_vocab(maps[src])  # 按照地址读vocab进去
    tok_src = Tokenizer(language=src, vocab=vocab_src,
                        encoding_length=args.MAX_INPUT_LENGTH)  # 然后初始化tokenizer类，这个类在utils函数中，可以完成词表，encode等等的操作
    vocab_tgt = read_vocab(maps[tgt])  # tgt同理
    tok_tgt = Tokenizer(language=tgt, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size src/tgt:{}/{}'.format(len(vocab_src), len(vocab_tgt)))  # 写log

    ## 构造training, validation, and testing dataloaders，这个在dataloader.py中，得到对应batch的数据
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_src, tok_tgt), args.batch_size,
                                                                 args.MAX_VID_LENGTH, (src, tgt), num_workers=0,
                                                                 pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format(len(train_loader), len(val_loader), len(test_loader)))  # 写log


# ----------------------------------------------------------------------------------------------------------
    vid_embed = None
    encoder = None
    decoder = None
    encoder_ch = None
    ## 初始化模型
    if args.model_type == 's2s':  # seq2seq，不过目前似乎只有这一种type

        vid_embed = Encoder_vid_embed(frame_size=args.frame_size, frame_embed_size=args.frame_embed_size).to(device)

        encoder_ch = Encoder_ch(vocab_size=len(vocab_src), word_embed_size=args.wordembed_dim, hidden_size=args.enc_ch_hid_size,
                                n_layers=1, dropout=0.2).to(device)

        encoder = Encoder_rnn_vid(frame_size=args.frame_size, frame_embed_size=args.frame_embed_size, hidden_size=args.enc_hid_size,
                          n_layers=1, dropout=0.2).to(device)
        decoder = Decoder_command(frame_embed_size=args.frame_embed_size, word_embed_size=args.wordembed_dim, hidden_size=args.dec_hid_size,
                          vocab_size=len(vocab_tgt),
                          n_layers=1, dropout=0.2, ps_threshold=args.PS_threshold, bias_vector=word_bias_vector_en).to(device)


    # 开始训练
    vid_embed.train()
    encoder.train()
    encoder_ch.train()
    decoder.train()

    ## loss是交叉熵
    # criterion = nn.CrossEntropyLoss(weight=word_weight_en, ignore_index=padding_idx, label_smoothing=0.05).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, label_smoothing=0.05).to(device)
    criterion_action = nn.CrossEntropyLoss(weight=action_weight, label_smoothing=0.01).to(device)
    ## 优化器都是Adam
    # dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
    #                                  lr=args.decoder_lr, weight_decay=args.weight_decay)
    # enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
    #                                  lr=args.encoder_lr, weight_decay=args.weight_decay)
    # vid_embed_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, vid_embed.parameters()),
    #                                  lr=args.encoder_lr, weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adamax(model.parameters(), lr=C.lr, weight_decay=1e-5)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, C.epochs, eta_min=0, last_epoch=-1)
# -------------------------------------------------------------------------------------------------------------------------
#     dec_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()),
#                                        lr=args.decoder_lr, weight_decay=args.weight_decay)
#     enc_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, encoder.parameters()),
#                                        lr=args.encoder_lr, weight_decay=args.weight_decay*50)
#     vid_embed_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, vid_embed.parameters()),
#                                              lr=args.vid_embed_lr, weight_decay=args.weight_decay)
#
#     enc_ch_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, encoder_ch.parameters()),
#                                        lr=args.encoder_ch_lr, weight_decay=args.weight_decay*1.0)
# -------------------------------------------------------------------------------------------------------------------------

    dec_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                       lr=args.decoder_lr, weight_decay=args.weight_decay)
    enc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                       lr=args.encoder_lr, weight_decay=args.weight_decay*1)
    vid_embed_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, vid_embed.parameters()),
                                             lr=args.vid_embed_lr, weight_decay=args.weight_decay)

    enc_ch_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_ch.parameters()),
                                       lr=args.encoder_ch_lr, weight_decay=args.weight_decay*1)




    ## 存loss
    total_train_loss, total_val_loss = [], []
    best_val_bleu, best_epoch = 0, 0
    best_bleu4 = 0
    scorer_sum_list = []
    scorer_bleu4_list = []

    action_acc_train, action_acc_val = [], []

    # resume = True
    resume = False
    if resume == True:  # 这里是从上一次最好的结果开始
    # 载入best参数
        init_epoch = torch.load(cp_file)['epoch']
        vid_embed.load_state_dict(torch.load(cp_file)['vid_embed_state_dict'])
        encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
        decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])
        encoder_ch.load_state_dict(torch.load(cp_file)['enc_ch_state_dict'])

        vid_embed_optimizer.load_state_dict(torch.load(cp_file)['vid_embed_optimizer'])
        enc_optimizer.load_state_dict(torch.load(cp_file)['enc_optimizer'])
        dec_optimizer.load_state_dict(torch.load(cp_file)['dec_optimizer'])
        enc_ch_optimizer.load_state_dict(torch.load(cp_file)['enc_ch_optimizer'])

        scorer_sum_list = torch.load(cp_file)['scorer_sum_list']
        scorer_bleu4_list = torch.load(cp_file)['scorer_bleu4_list']
        best_val_bleu = torch.load(cp_file)['best_val_bleu']
        best_bleu4 = torch.load(cp_file)['best_bleu4']

        action_acc_train = torch.load(cp_file)['action_acc_train']
        action_acc_val = torch.load(cp_file)['action_acc_val']

        total_train_loss = torch.load(cp_file)['total_train_loss']
        total_val_loss = torch.load(cp_file)['total_val_loss']


    # dec_lr_scheduler = torch.optim.lr_scheduler.StepLR(dec_optimizer, step_size=2, gamma=0.5)

    # dec_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(dec_optimizer, milestones=[100, 200], gamma=0.5)
    # enc_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(enc_optimizer, milestones=[100, 200], gamma=0.5)
    # vid_embed_optimizer_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(vid_embed_optimizer, milestones=[100, 200], gamma=0.5)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, C.epochs, eta_min=0, last_epoch=-1)

    dec_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dec_optimizer, args.epochs, eta_min=0.0001, last_epoch=-1)
    enc_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(enc_optimizer, args.epochs, eta_min=0.0001, last_epoch=-1)
    vid_embed_optimizer_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vid_embed_optimizer,
                                                                                  args.epochs, eta_min=0.0001, last_epoch=-1)

    enc_ch_optimizer_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(enc_ch_optimizer,
                                                                                  args.epochs, eta_min=0.0001, last_epoch=-1)

    # CosineAnnealingWarmRestarts

    count_paras(encoder, decoder, logging)  # 这里会打印一下总参数量



    ## 初始化时间
    zero_time = time.time()

    # 开始整个训练过程
    earlystop_flag = False  # 是否早停
    rising_count = 0

    eval_only = True
    # eval_only = False

    if not eval_only:
        for epoch in range(init_epoch, args.epochs):
            ## 开始按epoch迭代
            start_time = time.time()
            train_loss, action_test_acc_train = train(train_loader, vid_embed, encoder, encoder_ch, decoder, criterion, criterion_action, enc_ch_optimizer, vid_embed_optimizer, enc_optimizer, dec_optimizer,
                               epoch, args.epochs)  # 一个train周期，函数在下面

            # update the learning rate
            dec_lr_scheduler.step()
            enc_lr_scheduler.step()
            vid_embed_optimizer_lr_scheduler.step()
            enc_ch_optimizer_lr_scheduler.step()

            val_loss, sentbleu, corpbleu, final_results, nltk_result, action_test_acc_val = validate(val_loader, vid_embed, encoder, encoder_ch, decoder, criterion, criterion_action, tok_tgt, result_path, epoch)  # 一个验证周期，函数在下面
            scorer_sum = np.sum(final_results)
            scorer_sum_list.append(scorer_sum)
            scorer_bleu4_list.append(final_results[3]*10)  # bleu4

            # 存loss
            total_train_loss.append(train_loss)
            total_val_loss.append(val_loss)

            action_acc_train.append(action_test_acc_train)
            action_acc_val.append(action_test_acc_val)

            end_time = time.time()
            # 记录时间
            epoch_time = end_time - start_time
            total_time = end_time - zero_time

            logging.info(
                'Total time used: %s Epoch %d time uesd: %s train loss: %.4f val loss: %.4f sentbleu: %.4f corpbleu: %.4f' % (
                    str(datetime.timedelta(seconds=int(total_time))),
                    epoch, str(datetime.timedelta(seconds=int(epoch_time))), train_loss, val_loss, sentbleu, corpbleu))

            # if corpbleu > best_val_bleu:  # 更新最好的结果 scorer_sum
            #     best_val_bleu = corpbleu
            if scorer_sum > best_val_bleu:  # 更新最好的结果 scorer_sum
                best_val_bleu = scorer_sum

                checkpoint = {
                                'epoch': epoch + 1,
                                'vid_embed_state_dict': vid_embed.state_dict(),
                                'enc_state_dict': encoder.state_dict(),
                                'dec_state_dict': decoder.state_dict(),

                                'enc_ch_state_dict': encoder_ch.state_dict(),
                             }
                # torch.save(checkpoint, checkpoint_path + 'best_model_{}.pth'.format(epoch + 1))
                torch.save(checkpoint, checkpoint_path + 'best_model_min.pth')

                save_checkpoint({'epoch': epoch + 1,
                                 'vid_embed_state_dict': vid_embed.state_dict(),
                                 'enc_state_dict': encoder.state_dict(),
                                 'dec_state_dict': decoder.state_dict(),
                                 'enc_ch_state_dict': encoder_ch.state_dict(),

                                 'vid_embed_optimizer': vid_embed_optimizer.state_dict(),
                                 'enc_optimizer': enc_optimizer.state_dict(),
                                 'dec_optimizer': dec_optimizer.state_dict(),
                                 'enc_ch_optimizer': enc_ch_optimizer.state_dict(),

                                 'scorer_sum_list': scorer_sum_list, 'scorer_bleu4_list': scorer_bleu4_list,
                                 'best_val_bleu': best_val_bleu,
                                 'total_train_loss': total_train_loss, 'total_val_loss': total_val_loss,
                                 'action_acc_train': action_acc_train, 'action_acc_val': action_acc_val,
                                 }, cp_file)
                # best_epoch = epoch + 1

            # if scorer_sum > best_val_bleu:  # 更新最好的结果 scorer_sum
            #     best_val_bleu = scorer_sum
            if final_results[3] > best_bleu4:  # 更新最好的结果 scorer_sum
                best_bleu4 = final_results[3]

                checkpoint = {
                                'epoch': epoch + 1,
                                'vid_embed_state_dict': vid_embed.state_dict(),
                                'enc_state_dict': encoder.state_dict(),
                                'dec_state_dict': decoder.state_dict(),

                                'enc_ch_state_dict': encoder_ch.state_dict(),
                             }
                # torch.save(checkpoint, checkpoint_path + 'best_model_{}.pth'.format(epoch + 1))
                torch.save(checkpoint, checkpoint_path + 'best_bleu4_min.pth')

                save_checkpoint({'epoch': epoch + 1,
                                 'vid_embed_state_dict': vid_embed.state_dict(),
                                 'enc_state_dict': encoder.state_dict(),
                                 'dec_state_dict': decoder.state_dict(),
                                 'enc_ch_state_dict': encoder_ch.state_dict(),

                                 'vid_embed_optimizer': vid_embed_optimizer.state_dict(),
                                 'enc_optimizer': enc_optimizer.state_dict(),
                                 'dec_optimizer': dec_optimizer.state_dict(),
                                 'enc_ch_optimizer': enc_ch_optimizer.state_dict(),

                                 'scorer_sum_list': scorer_sum_list, 'scorer_bleu4_list': scorer_bleu4_list,
                                 'best_bleu4': best_bleu4,
                                 'total_train_loss': total_train_loss, 'total_val_loss': total_val_loss,
                                 'action_acc_train': action_acc_train, 'action_acc_val': action_acc_val,
                                 }, checkpoint_path + 'best_bleu4.pth')
                # best_epoch = epoch + 1


            logging.info("Finished {0} epochs of training".format(epoch + 1))  # 写log

            # # 存loss
            # total_train_loss.append(train_loss)
            # total_val_loss.append(val_loss)

            plot_epoch_loss(total_train_loss, total_val_loss, result_path)
            # plot_epoch_eval_result_total(scorer_sum_list, result_path)
            # plot_epoch_bleu4(scorer_bleu4_list, result_path)
            plot_epoch_result_total_bleu4(scorer_sum_list, scorer_bleu4_list, result_path)
            plot_epoch_action_acc(action_acc_train, action_acc_val, result_path)



    ### 最好效果的模型会被存起来，之后test的时候可以用
    logging.info('************ Start eval... ************')
    eval(test_loader, vid_embed, encoder, encoder_ch, decoder, cp_file, tok_tgt, result_path)


def train(train_loader, vid_embed, encoder, encoder_ch, decoder, criterion, criterion_action, enc_ch_optimizer, vid_embed_optimizer, enc_optimizer, dec_optimizer, epoch, EPOCH_NUMBER):
    '''
    执行每个eopch的训练
    '''
    vid_embed.train()
    encoder.train()
    decoder.train()
    encoder_ch.train()

    avg_loss = 0
    action_epoch_acc = 0
    for cnt, (srccap, tgtcap, video, img_key, img_key_neg, caplen_src, caplen_tgt, srcrefs, tgtrefs, action_class_label, srccap_half, caplen_src_half) in enumerate(train_loader, 1):
        # loader可以对应看dataloarder.py，cap是词表了，video是视频特征，caplen是长度方便索引，ref是无<PAD>的真实句子用于计算loss
        srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.long().to(device), tgtcap.long().to(device), video.to(device), caplen_src.to(device), caplen_tgt.to(device)
        img_key = img_key.to(device)
        img_key_neg = img_key_neg.to(device)
        action_class_label = action_class_label.long().to(device)

        # vid_out          = vid_embed(video[:, :, :2048])  # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
        # vid_embedded_neg = vid_embed(video_neg[:, :, :2048])

        vid_out          = vid_embed(img_key)  # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
        vid_embedded_neg = vid_embed(img_key_neg)
        # init_hidden, vid_out_rnn = encoder(video)

        (_, _), vid_out_rnn, act_class = encoder(video)
        src_out, init_hidden = encoder_ch(srccap)  # init_hidden  (_, _)

        action_epoch_acc += ((act_class.argmax(axis=1)) == action_class_label).sum().item()  # 累加acc，作为下面求平均acc的分子


        if (epoch + 1) < 30:
            teacher_forcing_ratio = 1.0
        elif (epoch + 1) >= 30 and (epoch + 1) < 60:
            teacher_forcing_ratio = 0.7
        elif (epoch + 1) >= 60 and (epoch + 1) < 120:
            teacher_forcing_ratio = 0.5
        else:
            teacher_forcing_ratio = 0.3


        scores, contrastive_attention = decoder(srccap, tgtcap, init_hidden, vid_out, vid_out_rnn, src_out, args.MAX_INPUT_LENGTH,
                         # vid_embedded_neg.detach_(),
                         vid_embedded_neg,
                         teacher_forcing_ratio=teacher_forcing_ratio)

        targets = tgtcap[:, 1:]  # 所有的句子从<start>开始，所以列表从1开始
        loss_cap = criterion(scores[:, 1:].contiguous().view(-1, decoder.vocab_size),
                         targets.contiguous().view(-1).long())  # 得到loss
        contrastive_attention_loss = F.binary_cross_entropy_with_logits(
            contrastive_attention.mean(dim=0), torch.FloatTensor([1, 0]).cuda())

        loss_action = criterion_action(act_class, action_class_label)  # 计算loss

        # loss = loss_cap + 0.1*contrastive_attention_loss + 0.05*loss_action
        loss = loss_cap

        # 反向传播
        dec_optimizer.zero_grad()  # 梯度清零
        if enc_optimizer is not None:
            enc_optimizer.zero_grad()
        vid_embed_optimizer.zero_grad()
        enc_ch_optimizer.zero_grad()

        loss.backward()

        # 梯度裁剪
        if args.grad_clip is not None:
            clip_gradient(dec_optimizer, args.grad_clip)
            clip_gradient(enc_optimizer, args.grad_clip)
            clip_gradient(vid_embed_optimizer, args.grad_clip)
            clip_gradient(enc_ch_optimizer, args.grad_clip)

            # 应用梯度
        dec_optimizer.step()
        enc_optimizer.step()
        vid_embed_optimizer.step()
        enc_ch_optimizer.step()

        # 记录loss
        avg_loss += loss.item()

        lr = dec_optimizer.state_dict()['param_groups'][0]['lr']
        t_c, stop_f = time_consum.show_Time_consuming(startTime, 10)

        print(
        '|batch[{}/{}]|batch_loss{: .5f}|lr{: .10f}|loss_cap{: .5f}|loss_CA{: .5f}|loss_action{: .5f}|<-|Epoch[{}/{}]: '
        .format(cnt, len(train_loader), loss.item(), lr, loss_cap.item(), contrastive_attention_loss.item(), loss_action.item(),
                epoch + 1,
                EPOCH_NUMBER) + t_c, flush=True)

    train_loader_len = 7699
    action_test_acc = action_epoch_acc/train_loader_len

    return avg_loss / cnt, action_test_acc


# video_neg = video_neg.to(device)
# # 前向传播，可train类似
# init_hidden, vid_out, vid_embedded_neg = encoder(video, video_neg)
#
# scores, pred_lengths = decoder.inference(tgtcap, init_hidden, vid_out, args.MAX_INPUT_LENGTH,
#                                          vid_embedded_neg,
#                                          )

def validate(val_loader, vid_embed, encoder, encoder_ch, decoder, criterion, criterion_action, tok_tgt, result_path, epoch):
    '''
    Performs one epoch's validation.
    '''
    decoder.eval()  # eval mode (没有dropout和batchnorm)
    vid_embed.eval()
    if encoder is not None:
        encoder.eval()
    encoder_ch.eval()

    references = list()  # references (真正的caption) 用于计算BLEU-4
    hypotheses = list()  # hypotheses (预测的caption)

    avg_loss = 0
    action_epoch_acc = 0
    with torch.no_grad():
        # 逐Batches
        for cnt, (srccap, tgtcap, video, img_key, img_key_neg, caplen_src, caplen_tgt, srcrefs, tgtrefs, action_class_label, srccap_half, caplen_src_half) in enumerate(val_loader, 1):
            srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.long().to(device), tgtcap.long().to(device), video.to(device), caplen_src.to(device), caplen_tgt.to(device)
            img_key = img_key.to(device)
            img_key_neg = img_key_neg.to(device)
            action_class_label = action_class_label.long().to(device)
            srccap_half, caplen_src_half = srccap_half.long().to(device), caplen_src_half.long().to(device)

            # 前向传播，可train类似
            # init_hidden, vid_out, vid_embedded_neg = encoder(video, video_neg)

            # vid_out = vid_embed(video[:, :, :2048])  # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
            vid_out = vid_embed(img_key)  # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
            # vid_embedded_neg = vid_embed(video_neg)
            # init_hidden, vid_out_rnn = encoder(video)
            (_, _), vid_out_rnn, act_class = encoder(video)

            # 使用半句话测试
            src_out, init_hidden = encoder_ch(srccap_half)  # init_hidden  (_, _)


            action_epoch_acc += ((act_class.argmax(axis=1)) == action_class_label).sum().item()  # 累加acc，作为下面求平均acc的分子

            # src,
            # trg, init_hidden,
            #
            # pos_vis_feats,
            # vid_out_rnn,
            # src_out,
            # max_len,
            # teacher_forcing_ratio = 0

            scores, pred_lengths = decoder.inference(srccap_half, tgtcap, init_hidden, vid_out, vid_out_rnn, src_out, args.MAX_INPUT_LENGTH)

            targets = tgtcap[:, 1:]
            scores_copy = scores.clone()

            # 计算在val数据集上的loss
            loss_caption = criterion(scores[:, 1:].contiguous().view(-1, decoder.vocab_size), targets.contiguous().view(-1))

            loss_action = criterion_action(act_class, action_class_label)  # 计算loss

            # loss = loss_caption + 0.1*loss_action
            loss = loss_caption

            # 预测结果
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][1:pred_lengths[j]])  # 移除 pads and idx-0

            preds = temp_preds
            hypotheses.extend(preds)  # preds= [1,2,3]

            tgtrefs = [list(map(int, i.split())) for i in tgtrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]

            for r in tgtrefs:  # tag会有多个句子，所以要保持一致算bleu
                references.append([r])

            assert len(references) == len(hypotheses)  # 强制长度一致，即一一对应关系

            avg_loss += loss.item()

        # 计算评估指数
        avg_loss = avg_loss / cnt
        corpbleu = corpus_bleu(references, hypotheses)  # 计算真实和预测的bleu
        sentbleu = 0
        sentence_bleu_1 = 0
        sentence_bleu_2 = 0
        sentence_bleu_3 = 0
        sentence_bleu_4 = 0

        meteor_nltk = 0

        for i, (r, h) in enumerate(zip(references, hypotheses), 1):
            # sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
            # sentbleu += sentence_bleu(r, h)
            sentence_bleu_1 += sentence_bleu(r, h, weights=(1, 0, 0, 0))
            sentence_bleu_2 += sentence_bleu(r, h, weights=(0, 1, 0, 0))
            sentence_bleu_3 += sentence_bleu(r, h, weights=(0, 0, 1, 0))
            sentence_bleu_4 += sentence_bleu(r, h, weights=(0, 0, 0, 1))

            r = [tok_tgt.decode_sentence(r[0])]
            h = tok_tgt.decode_sentence(h)
            meteor_nltk += meteor_score(r, h)

        # sentbleu /= i
        sentence_bleu_1 /= i
        sentence_bleu_2 /= i
        sentence_bleu_3 /= i
        sentence_bleu_4 /= i
        meteor_nltk /= i

        sentence_bleu_1 = round(sentence_bleu_1, 3)
        sentence_bleu_2 = round(sentence_bleu_2, 3)
        sentence_bleu_3 = round(sentence_bleu_3, 3)
        sentence_bleu_4 = round(sentence_bleu_4, 3)

        meteor_nltk = round(meteor_nltk, 3)

        nltk_result = (sentence_bleu_1, sentence_bleu_2, sentence_bleu_3, sentence_bleu_4, meteor_nltk)
# --------------------------------------------------------------------------------------------------------
        preds_decode_sentence = [tok_tgt.decode_sentence(t) for t in hypotheses]  # 把标号从词表中decode为句子
        tgtrefs_decode_sentence = [tok_tgt.decode_sentence(t[0]) for t in references]  # 把标号从词表中decode为句子

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # if os.path.exists(os.path.join(result_path + 'prediction.txt')):
        #     os.remove(os.path.join(result_path + 'prediction.txt'))

        f = open(os.path.join(result_path + 'prediction_{}.txt'.format(epoch+1)), 'w', encoding='utf-8')
        # f = open(os.path.join(result_path + 'prediction.txt'), 'w', encoding='utf-8')
        # if not os.path.exists(TRAIN_VOCAB_EN):

        for i in range(len(preds_decode_sentence)):
            # print(y_pred[i])
            # pred_command = utils.sequence_to_text(y_pred[i], vocab)
            pred_command = preds_decode_sentence[i]
            # print(y_true[i])
            # true_command = utils.sequence_to_text(y_true[i], vocab)
            true_command = tgtrefs_decode_sentence[i]
            f.write('------------------------------------------\n')
            f.write(str(i) + '\n')
            # f.write(pred_command + '\n')
            # f.write(true_command + '\n')
            f.write(true_command + '\n')
            f.write(pred_command + '\n')
        f.close()

        print('Ready for cococaption--Epoch[{}]'.format(epoch + 1), flush=True)

        final_results = val_iit_v2c(os.path.join(result_path + 'prediction_{}.txt'.format(epoch+1)))

        f = open(os.path.join(result_path + 'prediction_{}.txt'.format(epoch+1)), 'a+', encoding='utf-8')
        f.write('\n'*2)
        f.write('=============final_results===============' + '\n')
        f.write('Epoch =' + str(epoch+1) + '\n')

        val_loader_len = 3301
        action_test_acc = action_epoch_acc/val_loader_len
        f.write('Action Acc:{: .3f}\n'.format(action_test_acc))
        print('Action Acc:{: .3f}\n'.format(action_test_acc))

        for i in range(len(METRICS)):
            f.write('%s: %0.3f \n' % (METRICS[i], final_results[i]))

        METRICS_nltk= ["sentence_bleu_1", "sentence_bleu_2", "sentence_bleu_3", "sentence_bleu_4", "meteor_nltk"]
        for i in range(len(METRICS_nltk)):
            f.write('%s: %0.3f \n' % (METRICS_nltk[i], nltk_result[i]))

        print(' ')
        for i in range(len(METRICS_nltk)):
            print('%s: %0.3f' % (METRICS_nltk[i], nltk_result[i]))

        f.close()

    return avg_loss, sentbleu, corpbleu, final_results, nltk_result, action_test_acc


def eval(test_loader, vid_embed, encoder, encoder_ch, decoder, cp_file, tok_tgt, result_path):
    '''
    测试模型
    '''
    ### 会用最好的模型来测试
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))  # 写log
    # 载入best参数
    encoder_ch.load_state_dict(torch.load(cp_file)['enc_ch_state_dict'])
    vid_embed.load_state_dict(torch.load(cp_file)['vid_embed_state_dict'])
    encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
    decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])

    decoder.eval()  # eval mode (同样没有dropout or batchnorm)
    vid_embed.eval()
    if encoder is not None:
        encoder.eval()
    encoder_ch.eval()

    ids = list()  # 句子id
    hypotheses = list()  # hypotheses (预测的caption)
    references = list()  # references (真正的caption) 用于计算BLEU-4

    action_epoch_acc = 0
    with torch.no_grad():
        # 按Batches
        # for cnt, (srccap, video, caplen_src, sent_id) in enumerate(test_loader, 1):
        for cnt, (srccap, tgtcap, video, img_key, img_key_neg, caplen_src, caplen_tgt, srcrefs, tgtrefs, action_class_label, srccap_half, caplen_src_half) in enumerate(test_loader, 1):

            # srccap, video, caplen_src = srccap.long().to(device), video.to(device), caplen_src.to(device)
            srccap, tgtcap, video, caplen_src, caplen_tgt = srccap.long().to(device), tgtcap.long().to(
                device), video.to(device), caplen_src.to(device), caplen_tgt.to(device)
            img_key = img_key.to(device)
            img_key_neg = img_key_neg.to(device)
            action_class_label = action_class_label.long().to(device)
            srccap_half, caplen_src_half = srccap_half.long().to(device), caplen_src_half.long().to(device)

            # 前向，跟前面的函数一致
            # vid_out = vid_embed(video[:, :, :2048])  # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
            vid_out = vid_embed(img_key)  # 特征需要满足encoder和decoder相同：(mb, encout_dim) = (mb, decoder_dim)
            # vid_embedded_neg = vid_embed(video_neg)
            # init_hidden, vid_out_rnn = encoder(video)
            (_, _), vid_out_rnn, act_class = encoder(video)

            # 使用半句话测试
            src_out, init_hidden = encoder_ch(srccap_half)  # init_hidden  (_, _)

            action_epoch_acc += ((act_class.argmax(axis=1)) == action_class_label).sum().item()  # 累加acc，作为下面求平均acc的分子

            # src, init_decoder_hidden, vis_feats, vid_out_rnn, src_out, max_len, width = 2
            preds, pred_lengths = decoder.beam_decoding(srccap_half, init_hidden, vid_out, vid_out_rnn, src_out, args.MAX_INPUT_LENGTH,
                                                        width=1)

            # 预测caption
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][1:pred_lengths[j]])  # 移除 pads and idx-0

            # preds = [tok_tgt.decode_sentence(t) for t in temp_preds]  # 把标号从词表中decode为句子
            preds = temp_preds
            hypotheses.extend(preds)  # preds= [[1,2,3], ... ]

            tgtrefs = [list(map(int, i.split())) for i in tgtrefs]  # tgtrefs = [[1,2,3], [2,4,3], [1,4,5,]]

            for r in tgtrefs:  # tag会有多个句子，所以要保持一致算bleu
                references.append([r])

            assert len(references) == len(hypotheses)  # 强制长度一致，即一一对应关系


        sentbleu = 0
        sentence_bleu_1 = 0
        sentence_bleu_2 = 0
        sentence_bleu_3 = 0
        sentence_bleu_4 = 0

        meteor_nltk = 0

        for i, (r, h) in enumerate(zip(references, hypotheses), 1):
            # sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
            # sentbleu += sentence_bleu(r, h)
            sentence_bleu_1 += sentence_bleu(r, h, weights=(1, 0, 0, 0))
            sentence_bleu_2 += sentence_bleu(r, h, weights=(0, 1, 0, 0))
            sentence_bleu_3 += sentence_bleu(r, h, weights=(0, 0, 1, 0))
            sentence_bleu_4 += sentence_bleu(r, h, weights=(0, 0, 0, 1))

            r = [tok_tgt.decode_sentence(r[0])]
            h = tok_tgt.decode_sentence(h)
            meteor_nltk += meteor_score(r, h)

        # sentbleu /= i
        sentence_bleu_1 /= i
        sentence_bleu_2 /= i
        sentence_bleu_3 /= i
        sentence_bleu_4 /= i

        meteor_nltk /= i

        sentence_bleu_1 = round(sentence_bleu_1, 3)
        sentence_bleu_2 = round(sentence_bleu_2, 3)
        sentence_bleu_3 = round(sentence_bleu_3, 3)
        sentence_bleu_4 = round(sentence_bleu_4, 3)

        meteor_nltk = round(meteor_nltk, 3)

        nltk_result = (sentence_bleu_1, sentence_bleu_2, sentence_bleu_3, sentence_bleu_4, meteor_nltk)
# --------------------------------------------------------------------------------------------------------

        preds_decode_sentence = [tok_tgt.decode_sentence(t) for t in hypotheses]  # 把标号从词表中decode为句子
        tgtrefs_decode_sentence = [tok_tgt.decode_sentence(t[0]) for t in references]  # 把标号从词表中decode为句子
# -------------------------------------------------------------------------------------------------------------------
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if os.path.exists(os.path.join(result_path + 'prediction_best.txt')):
            os.remove(os.path.join(result_path + 'prediction_best.txt'))

        f = open(os.path.join(result_path + 'prediction_best.txt'), 'w', encoding='utf-8')

        for i in range(len(preds_decode_sentence)):
            # print(y_pred[i])
            # pred_command = utils.sequence_to_text(y_pred[i], vocab)
            pred_command = preds_decode_sentence[i]
            # print(y_true[i])
            # true_command = utils.sequence_to_text(y_true[i], vocab)
            true_command = tgtrefs_decode_sentence[i]
            f.write('------------------------------------------\n')
            f.write(str(i) + '\n')
            # f.write(pred_command + '\n')
            # f.write(true_command + '\n')
            f.write(true_command + '\n')
            f.write(pred_command + '\n')

        f.close()
        print('Ready for cococaption--Epoch[{}]'.format(epoch), flush=True)

        from cocoeval_iit_v2c import val_iit_v2c

        final_results = val_iit_v2c(os.path.join(result_path + 'prediction_best.txt'))

        f = open(os.path.join(result_path + 'prediction_best.txt'), 'a+', encoding='utf-8')
        f.write('\n'*2)
        f.write('=============final_results===============' + '\n')
        f.write('Epoch =' + str(epoch) + '\n')

        val_loader_len = 3301
        action_test_acc = action_epoch_acc/val_loader_len
        f.write('Action Acc:{: .3f}\n'.format(action_test_acc))
        print('Action Acc:{: .3f}\n'.format(action_test_acc))

        for i in range(len(METRICS)):
            f.write('%s: %0.3f \n' % (METRICS[i], final_results[i]))

        METRICS_nltk= ["sentence_bleu_1", "sentence_bleu_2", "sentence_bleu_3", "sentence_bleu_4", "meteor_nltk"]
        for i in range(len(METRICS_nltk)):
            f.write('%s: %0.3f \n' % (METRICS_nltk[i], nltk_result[i]))

        f.close()

        for i in range(len(METRICS)):
            print('%s: %0.3f' % (METRICS[i], final_results[i]))

        print(' ')
        for i in range(len(METRICS_nltk)):
            print('%s: %0.3f' % (METRICS_nltk[i], nltk_result[i]))

    return final_results


def test_no_label(test_loader, encoder, decoder, cp_file, tok_tgt, result_path):
    '''
    测试模型
    '''
    ### 会用最好的模型来测试
    epoch = torch.load(cp_file)['epoch']
    logging.info('Use epoch {0} as the best model for testing'.format(epoch))  # 写log
    # 载入best参数
    encoder.load_state_dict(torch.load(cp_file)['enc_state_dict'])
    decoder.load_state_dict(torch.load(cp_file)['dec_state_dict'])

    decoder.eval()  # eval mode (同样没有dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    ids = list()  # 句子id
    hypotheses = list()  # hypotheses (预测的caption)

    with torch.no_grad():
        # 按Batches
        for cnt, (srccap, video, video_neg, caplen_src, sent_id) in enumerate(test_loader, 1):
            srccap, video, caplen_src = srccap.long().to(device), video.to(device), caplen_src.to(device)
            video_neg = video_neg.to(device)
            # 前向，跟前面的函数一致
            src_out, init_hidden, vid_out = encoder(srccap, video)
            preds, pred_lengths = decoder.beam_decoding(srccap, init_hidden, src_out, vid_out, args.MAX_INPUT_LENGTH,
                                                        beam_size=2)

            # 预测caption
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:pred_lengths[j]])  # 移除 pads and idx-0

            preds = [tok_tgt.decode_sentence(t) for t in temp_preds]  # 把标号从词表中decode为句子

            hypotheses.extend(preds)  # preds= [[1,2,3], ... ]

            ids.extend(sent_id)

    ## 存一个预测出来caption文件submission
    dc = dict(zip(ids, hypotheses))
    print(len(dc))

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + 'submission.json', 'w') as fp:
        json.dump(dc, fp)
    return dc


def Generate_pretrained_word_vector(args):

    # vocab = build_vocab(args.DATA_DIR, language='en')
    vocab = build_vocab(args.DATA_DIR, language='zh')

    word_to_index = {}
    for i, word in enumerate(vocab):  # 根据vocab得到词到索引的字典
        word_to_index[word] = i

    embedding_type = "word2vec"
    # embedding_type = "glove"
    if embedding_type == "word2vec":
        embedding_weights = get_word2vec(vocab, word_to_index)
    elif embedding_type == "glove":
        embedding_weights = get_glove_embedding(vocab, word_to_index)
    else:
        pass
    pass

# ----- 构建词向量 -------------
def get_glove_embedding(vocab, word_to_index):
    '''
    生成glove词向量
    :return: 根据词表生成词向量
    '''

    path = os.path.abspath('./data')

    if not os.path.exists(path+"/glove_embedding_mr.npy"):  # 如果已经保存了词向量，就直接读取
        if not os.path.exists(path+"/glove_word2vec.txt"):
            glove_file = datapath(path+'/glove.840B.300d.txt')
            # 指定转化为word2vec格式后文件的位置
            tmp_file = get_tmpfile(path+"/glove_word2vec.txt")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove2word2vec(glove_file, tmp_file)
        else:
            tmp_file = get_tmpfile(path+"/glove_word2vec.txt")
        print("Reading Glove Embedding...")
        wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        tmp = []
        for word, index in word_to_index.items():
            try:
                tmp.append(wvmodel.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        print(mean, std)
        vocab_size = len(vocab)  # self.n_vocab
        embed_size = 300
        embedding_weights = np.random.normal(mean, std, [vocab_size, embed_size])  # 正太分布初始化方法
        # for word, index in vocab.word2idx.items():
        for word, index in word_to_index.items():
            try:
                embedding_weights[index, :] = wvmodel.get_vector(word)
            except:
                pass
        np.save(path+"/glove_embedding_mr.npy", embedding_weights)  # 保存生成的词向量
    else:
        embedding_weights = np.load(path+"/glove_embedding_mr.npy")  # 载入生成的词向量

    return embedding_weights


def get_word2vec(vocab, word_to_index):
    '''
    生成word2vec词向量
    :return: 根据词表生成的词向量
    '''

    path = os.path.abspath('./data')

    if not os.path.exists(path+"/word2vec_embedding_zh.npy"):  # 如果已经保存了词向量，就直接读取
        print("Reading word2vec Embedding...")

        # zh: sgns.merge.word.bz2
        # en: GoogleNews-vectors-negative300.bin.gz
        # wvmodel = KeyedVectors.load_word2vec_format(path+"/GoogleNews-vectors-negative300.bin.gz", binary=True)
        wvmodel = KeyedVectors.load_word2vec_format(path + "/sgns.merge.word.bz2", binary=False, encoding="utf-8", unicode_errors="ignore")
        tmp = []
        # for word, index in vocab.word2idx.items():
        for word, index in word_to_index.items():
            try:
                tmp.append(wvmodel.get_vector(word))
            except:
                pass
        mean = np.mean(np.array(tmp))
        std = np.std(np.array(tmp))
        print(mean, std)
        vocab_size = len(vocab)  # self.n_vocab
        embed_size = 300
        embedding_weights = np.random.normal(mean, std, [vocab_size,embed_size])  # 正太分布初始化方法
        for word, index in word_to_index.items():
            try:
                embedding_weights[index, :] = wvmodel.get_vector(word)
            except:
                pass
        np.save(path+"/word2vec_embedding_zh.npy", embedding_weights)  # 保存生成的词向量
    else:
        embedding_weights = np.load(path+"/word2vec_embedding_zh.npy")  # 载入生成的词向量

    return embedding_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMT')  # 参数
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:  # 载入参数
        import yaml

        args = Arguments(yaml.load(fin))
    main(args)  # 开始work

    # Generate_pretrained_word_vector(args)
    pass
