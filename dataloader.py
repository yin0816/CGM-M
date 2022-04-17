# -*- coding: utf-8 -*-
'''
Created on Mar 23, 2021
@author: nakaizura
'''

import json
import numpy as np
import os
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# 载入视频和句子数据

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

position_enc = PositionalEncoding(d_hid=2048, n_position=32)


def load_video_features(fpath, max_length):
    feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
    # 载入视频特征。帧数少了要补0，多了要采样
    if feats.shape[0] < max_length:
        dis = max_length - feats.shape[0]
        feats = np.lib.pad(feats, ((0, dis), (0, 0)), 'constant', constant_values=0)
    elif feats.shape[0] > max_length:
        inds = sorted(random.sample(range(feats.shape[0]), max_length))
        feats = feats[inds]
    assert feats.shape[0] == max_length  # 保证一致
    return np.float32(feats)

def load_video_features_2d_3d(fpath2d, fpath3d):
    # feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
    # 载入视频特征。帧数少了要补0，多了要采样
    use_3duan = True
    # use_3duan = False

    if use_3duan:
        Xv_2d = np.load(fpath2d)
        Xv_3d = np.load(fpath3d)

        # Xv_2d_random_sample = sorted(random.sample(Xv_2d, 30))

        Xv_2d = torch.from_numpy(Xv_2d)
        Xv_3d = torch.from_numpy(Xv_3d)

        # Xv_2d = position_enc(Xv_2d).squeeze(0)

        Xv_2d_a = Xv_2d[0:10]
        Xv_2d_b = Xv_2d[10:20]
        Xv_2d_c = Xv_2d[20:30]

        Xv_3d_a = Xv_3d[0].repeat(10, 1)
        Xv_3d_b = Xv_3d[1].repeat(10, 1)
        Xv_3d_c = Xv_3d[2].repeat(10, 1)

        Xva = torch.cat([Xv_2d_a, Xv_3d_a], dim=1)
        Xvb = torch.cat([Xv_2d_b, Xv_3d_b], dim=1)
        Xvc = torch.cat([Xv_2d_c, Xv_3d_c], dim=1)

        Xv = torch.cat([Xva, Xvb, Xvc], dim=0)
    else:
        Xv_2d = np.load(fpath2d)
        Xv_3d = np.load(fpath3d)

        # Xv_2d_random_sample = sorted(random.sample(Xv_2d, 30))

        Xv_2d = torch.from_numpy(Xv_2d)
        Xv_3d = torch.from_numpy(Xv_3d)

        # Xv_2d = position_enc(Xv_2d).squeeze(0)

        Xv_2d_a = Xv_2d[0:15]
        Xv_2d_b = Xv_2d[15:]
        # Xv_2d_a = Xv_2d[0:16]
        # Xv_2d_b = Xv_2d[16:]

        Xv_3d_a = Xv_3d[0].repeat(15, 1)
        Xv_3d_b = Xv_3d[1].repeat(15, 1)
        # Xv_3d_a = Xv_3d[0].repeat(16, 1)
        # Xv_3d_b = Xv_3d[1].repeat(16, 1)

        Xva = torch.cat([Xv_2d_a, Xv_3d_a], dim=1)
        Xvb = torch.cat([Xv_2d_b, Xv_3d_b], dim=1)

        Xv = torch.cat([Xva, Xvb], dim=0)

    return Xv

def load_video_features_2d_3d_key(fpath2d, fpath3d):
    # feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
    # 载入视频特征。帧数少了要补0，多了要采样

    use_3duan = True
    # use_3duan = False

    if use_3duan:
        Xv_2d = np.load(fpath2d)
        Xv_3d = np.load(fpath3d)

        # Xv_2d_random_sample = sorted(random.sample(Xv_2d, 30))

        Xv_2d = torch.from_numpy(Xv_2d)
        Xv_3d = torch.from_numpy(Xv_3d)

        # Xv_2d = position_enc(Xv_2d).squeeze(0)

        Xv_2d_a = Xv_2d[1:11]  # 删除首尾两帧
        Xv_2d_b = Xv_2d[11:21]
        Xv_2d_c = Xv_2d[21:31]


        Xv_3d_a = Xv_3d[0].repeat(10, 1)
        Xv_3d_b = Xv_3d[1].repeat(10, 1)
        Xv_3d_c = Xv_3d[2].repeat(10, 1)


        Xva = torch.cat([Xv_2d_a, Xv_3d_a], dim=1)
        Xvb = torch.cat([Xv_2d_b, Xv_3d_b], dim=1)
        Xvc = torch.cat([Xv_2d_c, Xv_3d_c], dim=1)

        Xv = torch.cat([Xva, Xvb, Xvc], dim=0)
    else:
        Xv_2d = np.load(fpath2d)
        Xv_3d = np.load(fpath3d)

        # Xv_2d_random_sample = sorted(random.sample(Xv_2d, 30))

        Xv_2d = torch.from_numpy(Xv_2d)
        Xv_3d = torch.from_numpy(Xv_3d)

        # Xv_2d = position_enc(Xv_2d).squeeze(0)

        Xv_2d_a = Xv_2d[1:16]  # 删除首尾两帧
        Xv_2d_b = Xv_2d[16:31]
        # Xv_2d_a = Xv_2d[0:16]
        # Xv_2d_b = Xv_2d[16:]

        Xv_3d_a = Xv_3d[0].repeat(15, 1)
        Xv_3d_b = Xv_3d[1].repeat(15, 1)
        # Xv_3d_a = Xv_3d[0].repeat(16, 1)
        # Xv_3d_b = Xv_3d[1].repeat(16, 1)

        Xva = torch.cat([Xv_2d_a, Xv_3d_a], dim=1)
        Xvb = torch.cat([Xv_2d_b, Xv_3d_b], dim=1)

        Xv = torch.cat([Xva, Xvb], dim=0)

    return Xv

def load_video_features_2d_3d_del(fpath2d_32f, fpath3d_32f, fpath2d, fpath3d):
    # feats = np.load(fpath, encoding='latin1')[0]  # encoding='latin1' to handle the inconsistency between python 2 and 3
    # 载入视频特征。帧数少了要补0，多了要采样

    if random.random() < 0.5:
        Xv_2d = np.load(fpath2d_32f)
        Xv_2d_s = []
        Xv_2d = list(Xv_2d)
        Xv_2d_random_sample = sorted(random.sample(range(0, len(Xv_2d)), 30))
        for i in Xv_2d_random_sample:
            Xv_2d_s.append(torch.from_numpy(Xv_2d[i]))

        Xv_2d_s = torch.stack(Xv_2d_s, dim=0)  # shape = [32, 3, 224, 224]
        # Xv_2d = torch.from_numpy(Xv_2d)
        Xv_2d = Xv_2d_s
    else:
        Xv_2d = np.load(fpath2d)
        Xv_2d = torch.from_numpy(Xv_2d)
# -----------------------------------------------------
    if random.random() < 0.5:
        Xv_3d = np.load(fpath3d_32f)
        Xv_3d = torch.from_numpy(Xv_3d)
    else:
        Xv_3d = np.load(fpath3d)
        Xv_3d = torch.from_numpy(Xv_3d)


# -----------------------------------------------------
    # Xv_2d = position_enc(Xv_2d).squeeze(0)

    Xv_2d_a = Xv_2d[0:15]
    Xv_2d_b = Xv_2d[15:]
    # Xv_2d_a = Xv_2d[0:16]
    # Xv_2d_b = Xv_2d[16:]

    Xv_3d_a = Xv_3d[0].repeat(15, 1)
    Xv_3d_b = Xv_3d[1].repeat(15, 1)
    # Xv_3d_a = Xv_3d[0].repeat(16, 1)
    # Xv_3d_b = Xv_3d[1].repeat(16, 1)

    Xva = torch.cat([Xv_2d_a, Xv_3d_a], dim=1)
    Xvb = torch.cat([Xv_2d_b, Xv_3d_b], dim=1)

    Xv = torch.cat([Xva, Xvb], dim=0)

    return Xv


class vatex_dataset(Dataset):
    # 载入文本的特征
    def __init__(self, data_dir, file_path, img_dir, split_type, tokenizers, max_vid_len, pair):
        src, tgt = pair
        maps = {'en': 'enCap', 'zh': 'chCap'}
        self.data_dir = data_dir
        self.img_dir = img_dir
        # tokenizer类，在utils.py中
        self.tok_src, self.tok_tgt = tokenizers
        self.max_vid_len = max_vid_len
        self.split_type = split_type

        with open(self.data_dir + file_path, 'r') as file:  # 打开数据文件
            data = json.load(file)
        self.srccaps, self.tgtcaps = [], []
        self.srcactions, self.tgtactions = [], []
        self.sent_ids = []
        for d in data:
            # srccap = d[maps[src]][5:]  # 词表
            srccap = d[maps[src]][0:]  # caption列表, 每个视频只有一个标签
            self.srccaps.extend(srccap)
            sent_id = [''.join((d['videoID'], '&', str(i))) for i in range(len(srccap))]  # 句子的id
            self.sent_ids.extend(sent_id)

            srcaction = d['action'][0:]
            self.srcactions.extend(srcaction)

            if split_type != 'test-no-label':
                # tgtcap = d[maps[tgt]][5:]
                tgtcap = d[maps[tgt]][0:]
                self.tgtcaps.extend(tgtcap)
        # ---------------------------------------------------
        # read class_indict
        json_file = 'data/action_class.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r',)
        self.class_dict = json.load(json_file)

        # load_negative_vids
        if self.split_type == 'train':
            with open('data/neg_vids_train.json', 'r') as fin:
                self.vid2neg_vids = json.load(fin)
        else:
            with open('data/neg_vids_test.json', 'r') as fin:
                self.vid2neg_vids = json.load(fin)

        # 构建配对的负样本
        self.negcaps = []
        self.sent_ids_neg = []
        for pos_vid in self.sent_ids:
            neg_vids = self.vid2neg_vids[pos_vid[:-2]]
            neg_vid = random.choice(neg_vids)
            neg_vid = neg_vid + '&0'
            neg_vid_id = self.sent_ids.index(neg_vid)
            self.negcaps.append(self.tgtcaps[neg_vid_id])
            self.sent_ids_neg.append(neg_vid)
        pass

        del data
        del self.vid2neg_vids
        import gc
        gc.collect()
        # import torch
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.srccaps)

    def __getitem__(self, idx):

        action_str = self.srcactions[idx]
        action_class = self.class_dict[action_str]
        action_class = torch.as_tensor(action_class, dtype=torch.int64)
        # ----------------------------------------------------------
        str_srccap, sent_id, sent_id_neg = self.srccaps[idx], self.sent_ids[idx], self.sent_ids_neg[idx]
        vid = sent_id[:-2]
        vid_neg = sent_id_neg[:-2]
        # srccap, caplen_src = self.tok_src.encode_sentence(str_srccap)  # 句子变id
        srccap, caplen_src, srccap_half, caplen_src_half = self.tok_src.encode_sentence_del(str_srccap)  # 句子变id
        srcref = self.tok_src.encode_sentence_nopad_2str(str_srccap)
        # img = load_video_features(os.path.join(self.data_dir,'vatex_features/',self.img_dir,vid+'.npy'), self.max_vid_len)

        # if self.split_type == 'train':
        #     img = load_video_features_2d_3d_del(os.path.join(self.data_dir, 'Res152-32f', vid + '.npy'),
        #                                         os.path.join(self.data_dir, 'Res50-3D-32f', vid + '.npy'),
        #
        #                                         os.path.join(self.data_dir, 'resnet152', vid + '.npy'),
        #                                         os.path.join(self.data_dir, 'resnet50-3D', vid + '.npy'),
        #                                         )
        #
        #     img_neg = load_video_features_2d_3d_del(os.path.join(self.data_dir, 'Res152-32f', vid_neg + '.npy'),
        #                                         os.path.join(self.data_dir, 'Res50-3D-32f', vid_neg + '.npy'),
        #
        #                                         os.path.join(self.data_dir, 'resnet152', vid_neg + '.npy'),
        #                                         os.path.join(self.data_dir, 'resnet50-3D', vid_neg + '.npy'),
        #                                         )
        #
        # else:
        #     img = load_video_features_2d_3d(os.path.join(self.data_dir, 'resnet152', vid + '.npy'),
        #                                     # os.path.join(self.data_dir, 'Res50-3D-KeyFrame', vid + '.npy'),
        #                                     os.path.join(self.data_dir, 'resnet50-3D', vid + '.npy'),
        #                                     )
        #
        #     img_neg = load_video_features_2d_3d(os.path.join(self.data_dir, 'resnet152', vid_neg + '.npy'),
        #                                     # os.path.join(self.data_dir, 'Res50-3D-KeyFrame', vid_neg + '.npy'),
        #                                     os.path.join(self.data_dir, 'resnet50-3D', vid_neg + '.npy'),
        #                                     )

        img = load_video_features_2d_3d(os.path.join(self.data_dir, 'resnet152', vid + '.npy'),
                                        # os.path.join(self.data_dir, 'Res50-3D-KeyFrame', vid + '.npy'),
                                        # os.path.join(self.data_dir, 'resnet50-3D', vid + '.npy'),
                                        os.path.join(self.data_dir, 'resnet50-3D-3duan', vid + '.npy'),
                                        )

        img_key = load_video_features_2d_3d_key(os.path.join(self.data_dir, 'Res152-KeyFrame', vid + '.npy'),
                                        # os.path.join(self.data_dir, 'Res50-3D-KeyFrame', vid + '.npy'),
                                        os.path.join(self.data_dir, 'Res50-3D-KeyFrame-3duan', vid + '.npy'),
                                        )

        img_key_neg = load_video_features_2d_3d_key(os.path.join(self.data_dir, 'Res152-KeyFrame', vid_neg + '.npy'),
                                        # os.path.join(self.data_dir, 'Res50-3D-KeyFrame', vid_neg + '.npy'),
                                        os.path.join(self.data_dir, 'Res50-3D-KeyFrame-3duan', vid_neg + '.npy'),
                                        )


        if self.split_type != 'test-no-label':  # 会要计算loss，所以会多一些参数
            str_tgtcap = self.tgtcaps[idx]
            # tgtcap, caplen_tgt = self.tok_tgt.encode_sentence(str_tgtcap)
            tgtcap, caplen_tgt, _, _ = self.tok_tgt.encode_sentence_del(str_tgtcap)
            tgtref = self.tok_tgt.encode_sentence_nopad_2str(str_tgtcap)
            return srccap, tgtcap, img, img_key, img_key_neg, caplen_src, caplen_tgt, srcref, tgtref, action_class, srccap_half, caplen_src_half
        else:
            return srccap, img, img_key, img_key_neg, caplen_src, sent_id, srccap_half, caplen_src_half


def get_loader(data_dir, tokenizers, split_type, batch_size, max_vid_len, pair, num_workers, pin_memory):
    # maps = {'train': ['vatex_training_v1.0.json', 'trainval'], 'val': ['vatex_validation_v1.0.json', 'trainval'],
    #         'test': ['vatex_public_test_english_v1.1.json', 'public_test']}

    maps = {'train': ['train-en-ch.json', 'Res152-32f'], 'val': ['test-en-ch.json', 'resnet152'],
            'test': ['test-en-ch.json', 'resnet152']}

    # maps = {'train': ['train-en-ch.json', 'Res152-KeyFrame'], 'val': ['test-en-ch.json', 'Res152-KeyFrame'],
    #         'test': ['test-en-ch.json', 'Res152-KeyFrame']}

    file_path, img_dir = maps[split_type]
    mydata = vatex_dataset(data_dir, file_path, img_dir, split_type, tokenizers, max_vid_len, pair)
    if split_type in ['train']:  # 打乱训练
        shuffle = True
    elif split_type in ['val', 'test']:
        shuffle = False
    else:
        shuffle = False
    # 载入loader
    myloader = DataLoader(dataset=mydata, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=pin_memory)
    return myloader


def create_split_loaders(data_dir, tokenizers, batch_size, max_vid_len, pair, num_workers=0, pin_memory=False):
    # 分别载入三者
    train_loader = get_loader(data_dir, tokenizers, 'train', batch_size, max_vid_len, pair, num_workers, pin_memory)
    val_loader = get_loader(data_dir, tokenizers, 'val', batch_size, max_vid_len, pair, num_workers, pin_memory)
    test_loader = get_loader(data_dir, tokenizers, 'test', batch_size, max_vid_len, pair, num_workers, pin_memory)
    # test_loader = [0]

    return train_loader, val_loader, test_loader
