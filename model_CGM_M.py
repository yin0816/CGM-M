'''
Created on Mar 23, 2021
@author: nakaizura
'''

import math
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F

from utils import sos_idx, eos_idx
from transformer.Models import Encoder as PhraseEncoder
from attention import SemanticAlignment, SemanticAttention, CrossAttention

# 模型的架构是s2s的，encode是LSTM，decode也是LSTM。
# 为了融合video，所以在输到decode的时候会融合当前词+源句子+视频特征的注意力。

phrase_encoder = PhraseEncoder(
    len_max_seq=10,  # config.loader.max_caption_len + 2,
    d_word_vec=300,  # config.vocab.embedding_size,
    n_layers=1,  # config.phr_encoder.SA_num_layers,
    n_head=1,  # config.phr_encoder.SA_num_heads,
    d_k=32,  # config.phr_encoder.SA_dim_k,
    d_v=32,  # config.phr_encoder.SA_dim_v,
    d_model=300,  # config.vocab.embedding_size,
    d_inner=512,  # config.phr_encoder.SA_dim_inner,
    dropout=0.1,  # config.phr_encoder.SA_dropout
)


class SoftDotAttention(nn.Module):
    def __init__(self, dim_ctx, dim_h):
        '''初始化层注意力层'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim_h, dim_ctx, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, context, h, mask=None):
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # 计算attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        weighted_ctx = torch.bmm(attn3, context)  # batch x dim
        return weighted_ctx, attn


# ----------------- 不使用中文作为输入 -------------------------------------------

class Encoder_vid_embed(nn.Module):
    def __init__(self, frame_size, frame_embed_size=None):
        super(Encoder_vid_embed, self).__init__()

        self.frame_embed = nn.Linear(frame_size*2, frame_embed_size)  # 视频 embedding

    def forward(self, vid):

        vid_embedded = self.frame_embed(vid)
        # vid_embedded = F.relu(vid_embedded)

        vid_out = vid_embedded

        return vid_out

class Encoder_ch(nn.Module):
    def __init__(self, vocab_size, word_embed_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder_ch, self).__init__()


        # self.use_pretrained = False
        self.use_pretrained = True
        if self.use_pretrained:
            embedding_weights = np.load('./data/word2vec_embedding_zh.npy')  # 载入生成的词向量
            embedding_pretrained = torch.from_numpy(embedding_weights).float()
            self.src_embed = nn.Embedding.from_pretrained(embedding_pretrained, freeze=True)
        else:
            self.src_embed = nn.Embedding(vocab_size, word_embed_size)  # 句子embedding

        self.src_encoder = nn.LSTM(input_size=word_embed_size, hidden_size=hidden_size // 2, num_layers=n_layers,
                                   dropout=dropout, batch_first=True, bidirectional=True)  # 多层LSTM


        self.dropout = nn.Dropout(dropout, inplace=True)

        self.n_layers = n_layers

        self.reset_parameters()

    def forward(self, src, src_hidden=None):

        batch_size = src.size(0)

        src_embedded = self.src_embed(src)  # 嵌入
        if self.use_pretrained == False:
            src_embedded = self.dropout(src_embedded)

        src_out, src_states = self.src_encoder(src_embedded, src_hidden)  # 过LSTM得到隐层
        # 2层LSTM的结果
        src_h = src_states[0].permute(1, 0, 2).contiguous().view(
            batch_size, self.n_layers, -1).permute(1, 0, 2)  # 前两维换一下
        src_c = src_states[1].permute(1, 0, 2).contiguous().view(
            batch_size, self.n_layers, -1).permute(1, 0, 2)

        init_h = src_h
        init_c = src_c

        return src_out, (init_h, init_c)

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

        if self.use_pretrained == False:
            nn.init.uniform_(self.src_embed.weight.data, -0.05, 0.05)

class Encoder_rnn_vid(nn.Module):
    def __init__(self, frame_size, frame_embed_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder_rnn_vid, self).__init__()

        self.frame_embed = nn.Linear(frame_size*2, frame_embed_size)  # 视频embedding

        self.video_encoder = nn.LSTM(input_size=frame_embed_size, hidden_size=hidden_size // 2, num_layers=n_layers,
                                     dropout=dropout, batch_first=True, bidirectional=True)  # 多层LSTM

        # self.video_encoder = nn.LSTM(input_size=frame_embed_size, hidden_size=hidden_size, num_layers=n_layers,
        #                              dropout=dropout, batch_first=True, bidirectional=True)  # 多层LSTM

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.n_layers = n_layers

        # self.fc = nn.Linear(hidden_size, 45)  # action_class = 44+1  # 不存在背景类
        self.fc = nn.Sequential(
                        # nn.Linear(hidden_size, hidden_size),
                        # nn.ReLU(True),
                        nn.Dropout(0.85),  # 0.5 0.75 0.85
                        nn.Linear(hidden_size, 45),
                      )

        self.reset_parameters()

    def forward(self, vid, vid_hidden=None):

        batch_size = vid.size(0)

        vid_embedded = self.frame_embed(vid)
        vid_embedded = F.relu(vid_embedded)

        vid_out, vid_states = self.video_encoder(vid_embedded, vid_hidden)  ##过LSTM得到隐层

        # 1层LSTM的结果
        vid_h = vid_states[0].permute(1, 0, 2).contiguous().view(
            batch_size, self.n_layers, -1).permute(1, 0, 2)
        vid_c = vid_states[1].permute(1, 0, 2).contiguous().view(
            batch_size, self.n_layers, -1).permute(1, 0, 2)

        init_h = vid_h
        init_c = vid_c

        # act_output = init_h[0]
        act_output = vid_out.mean(dim=1)  # axis=1
        act_class = self.fc(act_output)

        return (init_h, init_c), vid_out, act_class

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


class Decoder_command(nn.Module):
    def __init__(self, word_embed_size, frame_embed_size, hidden_size, vocab_size,
                 n_layers=2, dropout=0.5, ps_threshold=0.2, bias_vector=None):
        super(Decoder_command, self).__init__()
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.PS_threshold = ps_threshold

        # self.use_pretrained = False
        self.use_pretrained = True
        if self.use_pretrained:
            embedding_weights = np.load('./data/word2vec_embedding_en.npy')  # 载入生成的词向量
            embedding_pretrained = torch.from_numpy(embedding_weights).float()
            self.embed = nn.Embedding.from_pretrained(embedding_pretrained, freeze=True)
        else:
            self.embed = nn.Embedding(vocab_size, word_embed_size)  # embedding

        self.dropout = nn.Dropout(dropout, inplace=True)
        # self.src_attention = SoftDotAttention(word_embed_size, hidden_size)  # src的attention
        self.src_attention = SoftDotAttention(hidden_size, hidden_size)  # src的attention
        self.vid_attention = SoftDotAttention(hidden_size, hidden_size)  # video的attention

        # self.decoder = nn.LSTM((frame_embed_size+word_embed_size)+word_embed_size+hidden_size+hidden_size, hidden_size,
        #                        n_layers, dropout=dropout, batch_first=False)  # 3倍：当前词+源句子+视频

        self.decoder = nn.LSTM((0+0)+word_embed_size+hidden_size+hidden_size, hidden_size,
                               n_layers, dropout=dropout, batch_first=False)  # 3倍：当前词+源句子+视频

        # self.decoder = nn.LSTM((0+0)+word_embed_size+hidden_size, hidden_size,
        #                        n_layers, dropout=dropout, batch_first=False)  # 3倍：当前词+源句子+视频

        # crossattention
        # self.decoder = nn.LSTM((0+0)+word_embed_size+hidden_size, hidden_size,
        #                        n_layers, dropout=dropout, batch_first=False)  # 3倍：当前词+源句子+视频

        # self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.embed_size),
        #                         nn.Tanh(),
        #                         nn.Dropout(p=dropout),
        #                         nn.Linear(embed_size, vocab_size))

        self.fc = nn.Linear(hidden_size, vocab_size, bias=True)
        self.reset_parameters(bias_vector)

        self.phr_encoder = PhraseEncoder(
                                        len_max_seq=10,  # config.loader.max_caption_len + 2,
                                        d_word_vec=word_embed_size,  # config.vocab.embedding_size,
                                        n_layers=1,  # config.phr_encoder.SA_num_layers,
                                        n_head=1,  # config.phr_encoder.SA_num_heads,
                                        d_k=32,  # config.phr_encoder.SA_dim_k,
                                        d_v=32,  # config.phr_encoder.SA_dim_v,
                                        d_model=300,  # config.vocab.embedding_size,
                                        d_inner=512,  # config.phr_encoder.SA_dim_inner,
                                        dropout=0.1,  # config.phr_encoder.SA_dropout
                                    )
        self.semantic_alignment = SemanticAlignment(
            query_size=word_embed_size,  # self.embedding_size,
            feat_size=frame_embed_size,  # self.vis_feat_size,
            bottleneck_size=512,  # self.sem_align_hidden_size
            )
        self.semantic_attention = SemanticAttention(
            query_size=hidden_size,  # self.hidden_size,
            key_size=word_embed_size+frame_embed_size,  # self.embedding_size + self.vis_feat_size,
            bottleneck_size=512,  # self.sem_attn_hidden_size
        )
        # self.crossattention = CrossAttention(
        #     query_size=hidden_size,  # self.hidden_size,
        #     key_size=hidden_size,  # self.embedding_size + self.vis_feat_size,
        #     bottleneck_size=512,  # self.sem_attn_hidden_size
        # )

    def get_rnn_init_hidden(self, batch_size, num_layers, hidden_size):
        return (
            torch.zeros(num_layers, batch_size, hidden_size).cuda(),
            torch.zeros(num_layers, batch_size, hidden_size).cuda()
        )

    def onestep_sgn(self, embedded, hidden, vis_feats, vid_out_rnn, phr_feats, phr_masks,  src_out, src_mask):
        last_hidden = hidden

        semantic_group_feats, semantic_align_weights, semantic_align_logits = self.semantic_alignment(
            phr_feats=phr_feats,
            vis_feats=vis_feats)
        feat, semantic_attn_weights, semantic_attn_logits = self.semantic_attention(
            query=last_hidden[0][0],
            keys=semantic_group_feats,
            values=semantic_group_feats,
            masks=phr_masks)

        vid_ctx, vid_attn = self.vid_attention(vid_out_rnn, last_hidden[0][0])
        src_ctx, src_attn = self.src_attention(src_out, last_hidden[0][0], mask=src_mask)
        vid_ctx = vid_ctx.squeeze(1)
        src_ctx = src_ctx.squeeze(1)


        # rnn_input = torch.cat([embedded, vid_ctx], dim=1)
        # rnn_input = torch.cat((feat, embedded), dim=1)
        # rnn_input = torch.cat((feat, embedded, vid_ctx), dim=1)
        # rnn_input = torch.cat((feat, embedded, vid_ctx, src_ctx), dim=1)



        rnn_input = torch.cat((embedded, vid_ctx, src_ctx), dim=1)  # no-sgn

        # ---------------------------------------------------------------------------------------------
        # crossattention
        # rnn_input_cross_attention = self.crossattention(
        #                                 query=last_hidden[0][0],
        #                                 keys_v=vid_ctx,
        #                                 values_v=vid_ctx,
        #                                 keys_t=src_ctx,
        #                                 values_t=src_ctx)
        # rnn_input = torch.cat((embedded, rnn_input_cross_attention.squeeze(1)), dim=1)
        # ---------------------------------------------------------------------------------------------

        output, hidden = self.decoder(rnn_input[None, :, :], last_hidden)  # 再LSTM解码
        # output = output.squeeze(1)  # (B, 1, N) -> (B,N)
        output = output.squeeze(0)
        output_logits = self.fc(output)
        # output = torch.log_softmax(output_logits, dim=1)
        output = output_logits

        return output, output_logits, hidden, (semantic_align_weights, semantic_attn_weights), \
               (semantic_align_logits, semantic_attn_logits)

    def reset_parameters(self, bias_vector):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

        if self.use_pretrained == False:
            nn.init.uniform_(self.embed.weight.data, -0.05, 0.05)

        if bias_vector is not None:
            self.fc.bias.data = torch.from_numpy(bias_vector).float()

    def forward(self,
                src,
                trg, init_hidden,
                pos_vis_feats,
                vid_out_rnn,
                src_out,
                max_len,
                neg_vis_feats,
                teacher_forcing_ratio):
        # 普通搜索
        batch_size = trg.size(0)
        src_mask = (src == 0)  # mask paddings.
        trg_T = trg.T

        caption_EOS_table = trg_T == 3  # '<EOS>'
        caption_PAD_table = trg_T == 0  # '<PAD>'
        caption_end_table = ~(~caption_EOS_table * ~caption_PAD_table)

        # hidden = self.get_rnn_init_hidden(batch_size, self.n_layers, hidden_size=600)
        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).cuda()

        caption_lens = torch.zeros(batch_size).cuda().long()
        contrastive_attention_list = []
        # output = trg.data[:, 0]  # <sos>
        output = torch.LongTensor(1, batch_size).fill_(2).cuda() # <sos>
        for t in range(1, max_len):
            embedded = self.embed(output.view(1, -1)).squeeze(0)
            if t == 1:
                embedded_list = embedded[:, None, :]
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            elif t == 2:
                embedded_list = embedded[:, None, :]
                caption_lens += 1
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            else:
                embedded_list = torch.cat([embedded_list, embedded[:, None, :]], dim=1)
                caption_lens += ((output.long().squeeze() != 0) * \
                                 (output.long().squeeze() != 3)).long()
                src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                src_pos[src_pos > caption_lens[:, None]] = 0  # SelfAttention_PAD
            phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
            phr_attns = phr_attns[0]

            if t >= 2:
                A = torch.bmm(phr_attns, phr_attns.transpose(1, 2))
                A_mask = torch.eye(t-1, t-1).cuda().bool()
                A.masked_fill_(A_mask, 0)
                A_sum = A.sum(dim=2)

                indices = (A >= self.PS_threshold).nonzero()  # Obtain indices of phrase pairs that
                                                                             # are highly overlapped with each other
                indices = indices[indices[:, 1] < indices[:, 2]] # Leave only the upper triangle to prevent duplication

                phr_masks = torch.zeros_like(A_sum).bool()
                if len(indices) > 0:
                    redundancy_masks = torch.zeros_like(phr_masks).long()
                    indices_b = indices[:, 0]
                    indices_i = indices[:, 1]
                    indices_j = indices[:, 2]
                    indices_ij = torch.stack(( indices_i, indices_j ), dim=1)
                    A_sum_i = A_sum[indices_b, indices_i]
                    A_sum_j = A_sum[indices_b, indices_j]
                    A_sum_ij = torch.stack(( A_sum_i, A_sum_j ), dim=1)
                    _, i_or_j = A_sum_ij.max(dim=1)
                    i_or_j = i_or_j.bool()
                    indices_i_or_j = torch.zeros_like(indices_b)
                    indices_i_or_j[i_or_j] = indices_j[i_or_j]
                    indices_i_or_j[~i_or_j] = indices_i[~i_or_j]
                    redundancy_masks[indices_b, indices_i_or_j] = 1 # Mask phrases that are more redundant
                                                                    # than their counterpart
                    phr_masks = redundancy_masks > 0.5
            else:
                phr_masks = None

            output, output_logits, hidden, (sem_align_weights, _), (sem_align_logits, _) = self.onestep_sgn(
                embedded, hidden, pos_vis_feats, vid_out_rnn, phr_feats, phr_masks,  src_out, src_mask)  # (mb, vocab) (1, mb, N) (mb, 1, seqlen)

            # Calculate the Contrastive Attention loss
            if t >= 2:
                pos_sem_align_logits = sem_align_logits
                _, _, neg_sem_align_logits = self.semantic_alignment(phr_feats, neg_vis_feats)
                pos_align_logit = pos_sem_align_logits.sum(dim=2)
                neg_align_logit = neg_sem_align_logits.sum(dim=2)
                pos_align_logit = pos_align_logit[~caption_end_table[t-1]]
                neg_align_logit = neg_align_logit[~caption_end_table[t-1]]
                align_logits = torch.stack([ pos_align_logit, neg_align_logit ], dim=2)
                phr_masks_for_logits = phr_masks[~caption_end_table[t-1]]
                align_logits = align_logits.view(-1, 2)[~phr_masks_for_logits.view(-1)]
                contrastive_attention_list.append(align_logits)

            # Early stop
            if torch.all(caption_end_table[t]).item():
                break

            outputs[:, t, :] = output_logits
            is_teacher = random.random() < teacher_forcing_ratio  # 一定比例top1
            top1 = output.data.max(1)[1]
            # output = (trg.data[:, t] if is_teacher else top1).cuda()  # 选择输出作为下一次的当前词嵌入
            output = (trg_T.data[t] if is_teacher else top1).cuda()  # 选择输出作为下一次的当前词嵌入

        contrastive_attention = torch.cat(contrastive_attention_list, dim=0)
        return outputs, contrastive_attention

    def inference(self,
                  src,
                  trg, init_hidden,

                  pos_vis_feats,
                  vid_out_rnn,
                  src_out,
                  max_len,
                  teacher_forcing_ratio=0
                 ):
        # 贪心搜索
        batch_size = trg.size(0)
        src_mask = (src == 0)  # mask paddings.
        trg_T = trg.T

        # hidden = self.get_rnn_init_hidden(batch_size, self.n_layers, hidden_size=600)
        hidden = (init_hidden[0][:self.n_layers].contiguous(), init_hidden[1][:self.n_layers].contiguous())
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).cuda()

        caption_lens = torch.zeros(batch_size).cuda().long()

        pred_lengths = [0] * batch_size

        # output = trg.data[:, 0]  # <sos>
        output = torch.LongTensor(1, batch_size).fill_(2).cuda()  # <sos>
        for t in range(1, max_len):
            embedded = self.embed(output.view(1, -1)).squeeze(0)
            if t == 1:
                embedded_list = embedded[:, None, :]
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            elif t == 2:
                embedded_list = embedded[:, None, :]
                caption_lens += 1
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            else:
                embedded_list = torch.cat([embedded_list, embedded[:, None, :]], dim=1)
                caption_lens += ((output.long().squeeze() != 0) * \
                                 (output.long().squeeze() != 3)).long()
                src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                src_pos[src_pos > caption_lens[:, None]] = 0  # SelfAttention_PAD
            phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
            phr_attns = phr_attns[0]

            if t >= 2:
                A = torch.bmm(phr_attns, phr_attns.transpose(1, 2))
                A_mask = torch.eye(t - 1, t - 1).cuda().bool()
                A.masked_fill_(A_mask, 0)
                A_sum = A.sum(dim=2)

                indices = (A >= self.PS_threshold).nonzero()  # Obtain indices of phrase pairs that
                # are highly overlapped with each other
                indices = indices[indices[:, 1] < indices[:, 2]]  # Leave only the upper triangle to prevent duplication

                phr_masks = torch.zeros_like(A_sum).bool()
                if len(indices) > 0:
                    redundancy_masks = torch.zeros_like(phr_masks).long()
                    indices_b = indices[:, 0]
                    indices_i = indices[:, 1]
                    indices_j = indices[:, 2]
                    indices_ij = torch.stack((indices_i, indices_j), dim=1)
                    A_sum_i = A_sum[indices_b, indices_i]
                    A_sum_j = A_sum[indices_b, indices_j]
                    A_sum_ij = torch.stack((A_sum_i, A_sum_j), dim=1)
                    _, i_or_j = A_sum_ij.max(dim=1)
                    i_or_j = i_or_j.bool()
                    indices_i_or_j = torch.zeros_like(indices_b)
                    indices_i_or_j[i_or_j] = indices_j[i_or_j]
                    indices_i_or_j[~i_or_j] = indices_i[~i_or_j]
                    redundancy_masks[indices_b, indices_i_or_j] = 1  # Mask phrases that are more redundant
                    # than their counterpart
                    phr_masks = redundancy_masks > 0.5
            else:
                phr_masks = None

            output, output_logits, hidden, (sem_align_weights, _), (sem_align_logits, _) = self.onestep_sgn(
                embedded, hidden, pos_vis_feats, vid_out_rnn, phr_feats, phr_masks,  src_out, src_mask)  # (mb, vocab) (1, mb, N) (mb, 1, seqlen)

            # Calculate the Contrastive Attention loss

            # Early stop
            # if torch.all(caption_end_table[t]).item():
            #     break

            outputs[:, t, :] = output_logits
            is_teacher = random.random() < teacher_forcing_ratio  # 一定比例top1
            top1 = output.data.max(1)[1]
            # output = (trg.data[:, t] if is_teacher else top1).cuda()  # 选择输出作为下一次的当前词嵌入
            output = (trg_T.data[t] if is_teacher else top1).cuda()  # 选择输出作为下一次的当前词嵌入

            for i in range(batch_size):
                if output[i] == 3 and pred_lengths[i] == 0:
                    pred_lengths[i] = t
        for i in range(batch_size):
            if pred_lengths[i] == 0:
                pred_lengths[i] = max_len
        return outputs, pred_lengths

    def beam_decoding(self, src, init_decoder_hidden, vis_feats, vid_out_rnn, src_out, max_len, width=2):
        # 束搜索

        batch_size = vis_feats.size(0)
        vocab_size = self.vocab_size

        src_mask = (src == 0)  # mask padding

        # hidden = self.get_rnn_init_hidden(batch_size, self.n_layers, hidden_size=600)
        hidden = init_decoder_hidden

        input_list = [torch.cuda.LongTensor(1, batch_size).fill_(2)]  # '<SOS>'
        hidden_list = [hidden]
        cum_prob_list = [torch.ones(batch_size).cuda()]
        cum_prob_list = [torch.log(cum_prob) for cum_prob in cum_prob_list]
        EOS_idx = 3  # self.vocab.word2idx['<EOS>']

        output_list = [[[]] for _ in range(batch_size)]
        for t in range(1, max_len):
            beam_output_list = []
            normalized_beam_output_list = []
            beam_hidden_list = ([], [])
            next_output_list = [[] for _ in range(batch_size)]

            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                caption_list = [output_list[b][i] for b in range(batch_size)]
                if t == 1:
                    words_list = input.transpose(0, 1)
                else:
                    words_list = torch.cuda.LongTensor(caption_list)

                embedded_list = self.embed(words_list)
                if t == 1:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                elif t == 2:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                else:
                    caption_lens = torch.cuda.LongTensor([[idx.item() for idx in caption].index(EOS_idx) if EOS_idx in [
                        idx.item() for idx in caption] else t - 1 for caption in caption_list])
                    src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                    src_pos[src_pos > caption_lens[:, None]] = 0
                phr_feats, phr_attns = self.phr_encoder(embedded_list, src_pos, return_attns=True)
                phr_attns = phr_attns[0]

                if t >= 2:
                    A = torch.bmm(phr_attns, phr_attns.transpose(1, 2))
                    A_mask = torch.eye(t - 1, t - 1).cuda().bool()
                    A.masked_fill_(A_mask, 0)
                    A_sum = A.sum(dim=2)

                    indices = (A >= self.PS_threshold).nonzero()
                    indices = indices[
                        indices[:, 1] < indices[:, 2]]  # Leave only the upper triangle to prevent duplication

                    phr_masks = torch.zeros_like(A_sum).bool()
                    if len(indices) > 0:
                        redundancy_masks = torch.zeros_like(phr_masks).long()
                        indices_b = indices[:, 0]
                        indices_i = indices[:, 1]
                        indices_j = indices[:, 2]
                        indices_ij = torch.stack((indices_i, indices_j), dim=1)
                        A_sum_i = A_sum[indices_b, indices_i]
                        A_sum_j = A_sum[indices_b, indices_j]
                        A_sum_ij = torch.stack((A_sum_i, A_sum_j), dim=1)
                        _, i_or_j = A_sum_ij.max(dim=1)
                        i_or_j = i_or_j.bool()
                        indices_i_or_j = torch.zeros_like(indices_b)
                        indices_i_or_j[i_or_j] = indices_j[i_or_j]
                        indices_i_or_j[~i_or_j] = indices_i[~i_or_j]
                        redundancy_masks[indices_b, indices_i_or_j] = 1  # Mask phrases that are more redundant
                        # than their counterpart
                        phr_masks = redundancy_masks > 0.5
                else:
                    phr_masks = None

                embedded = self.embed(input.view(1, -1)).squeeze(0)
                # output, next_hidden, _, _ = self.decoder(embedded, hidden, vis_feats, phr_feats, phr_masks)
                output, output_logits, next_hidden, (sem_align_weights, _), (sem_align_logits, _) = self.onestep_sgn(
                    embedded, hidden, vis_feats, vid_out_rnn, phr_feats, phr_masks,  src_out, src_mask)  # (mb, vocab) (1, mb, N) (mb, 1, seqlen)

                EOS_mask = [1 if EOS_idx in [idx.item() for idx in caption] else 0 for caption in caption_list]
                EOS_mask = torch.cuda.BoolTensor(EOS_mask)
                output[EOS_mask] = 0.

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [[idx.item() for idx in caption].index(EOS_idx) + 1 if EOS_idx in [idx.item() for idx in
                                                                                                  caption] else t for
                                caption in caption_list]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** 1.6) / ((5 + 1) ** 1.6)
                normalized_output = output / normalizing_factor[:, None]
                normalized_beam_output_list.append(normalized_output)
                beam_hidden_list[0].append(next_hidden[0])
                beam_hidden_list[1].append(next_hidden[1])
            beam_output_list = torch.cat(beam_output_list, dim=1)  # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:,
                                          :width]  # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size  # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size  # ( 100, width )

            topk_output_list = [topk_output_index[:, i] for i in range(width)]  # width * ( 100, )
            topk_hidden_list = (
                [[] for _ in range(width)],
                [[] for _ in range(width)])  # 2 * width * (1, 100, 512)
            topk_cum_prob_list = [[] for _ in range(width)]  # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                    topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [oi])
            output_list = next_output_list

            input_list = [topk_output.unsqueeze(0) for topk_output in topk_output_list]  # width * ( 1, 100 )
            hidden_list = (
                [torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0]],
                [torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1]])  # 2 * width * ( 1, 100, 512 )
            hidden_list = [(hidden, context) for hidden, context in zip(*hidden_list)]
            cum_prob_list = [torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in
                             topk_cum_prob_list]  # width * ( 100, )

        SOS_idx = torch.tensor(2).cuda()  # self.vocab.word2idx['<SOS>']
        outputs = [[SOS_idx] + o[0] for o in output_list]

        outputs_list = [torch.stack(aa, dim=0) for aa in outputs]
        outputs_list_out = torch.cat((outputs_list), dim=0).view(-1, max_len)

        seq=outputs_list_out.cpu()

        pred_lengths = [0] * batch_size
        for i in range(batch_size):
            if sum(seq[i] == eos_idx) == 0:
                pred_lengths[i] = max_len
            else:
                pred_lengths[i] = (seq[i] == eos_idx).nonzero()[0][0]
        # return the samples and their log likelihoods
        return seq, pred_lengths  # seq_log_probs

    def from_numpy(self, states):
        return [torch.from_numpy(state).contiguous().cuda() for state in states]

    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        return np.unravel_index(args, matrix.shape), flatten[args]
