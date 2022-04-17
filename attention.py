import torch
import torch.nn as nn


class SemanticAlignment(nn.Module):
    def __init__(self, query_size, feat_size, bottleneck_size):
        super(SemanticAlignment, self).__init__()
        self.query_size = query_size
        self.feat_size = feat_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.feat_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, phr_feats, vis_feats):
        Wh = self.W(phr_feats)
        Uv = self.U(vis_feats)

        energies = self.w(torch.tanh(Wh[:, :, None, :] + Uv[:, None, :, :] + self.b)).squeeze(-1)
        weights = torch.softmax(energies, dim=2)
        aligned_vis_feats = torch.bmm(weights, vis_feats)
        semantic_group_feats = torch.cat([ phr_feats, aligned_vis_feats ], dim=2)
        return semantic_group_feats, weights, energies


class SemanticAttention(nn.Module):
    def __init__(self, query_size, key_size, bottleneck_size):
        super(SemanticAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U = nn.Linear(self.key_size, self.bottleneck_size, bias=False)
        self.b = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)

    def forward(self, query, keys, values, masks=None):
        Wh = self.W(query)
        Uv = self.U(keys)
        Wh = Wh.unsqueeze(1).expand_as(Uv)
        energies = self.w(torch.tanh(Wh + Uv + self.b))
        if masks is not None:
            masks = masks[:, :, None]
            energies[masks] = -float('inf')
        weights = torch.softmax(energies, dim=1)
        weighted_feats = values * weights.expand_as(values)
        attn_feats = weighted_feats.sum(dim=1)
        return attn_feats, weights, energies


class CrossAttention(nn.Module):
    def __init__(self, query_size, key_size, bottleneck_size):
        super(CrossAttention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.bottleneck_size = bottleneck_size

        self.W = nn.Linear(self.query_size, self.bottleneck_size, bias=False)
        self.U_v = nn.Linear(self.key_size, self.bottleneck_size, bias=False)
        self.Q_v = nn.Linear(self.key_size, self.key_size, bias=False)
        # self.b_v = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.U_t = nn.Linear(self.key_size, self.bottleneck_size, bias=False)
        self.Q_t = nn.Linear(self.key_size, self.key_size, bias=False)
        # self.b_t = nn.Parameter(torch.ones(self.bottleneck_size), requires_grad=True)
        self.w = nn.Linear(self.bottleneck_size, 1, bias=False)
        # self.w = nn.Linear(self.bottleneck_size, self.query_size, bias=False)

    def forward(self, query, keys_v, values_v, keys_t, values_t):
        Wh = self.W(query)

        U_v = self.U_v(keys_v)
        U_t = self.U_t(keys_t)

        Wh_v = Wh
        # energies_v = self.w(torch.tanh(Wh_v + U_v + self.b_v))
        energies_v = self.w(torch.tanh(Wh_v + U_v))

        Wh_t = Wh
        # energies_t = self.w(torch.tanh(Wh_t + U_t + self.b_t))
        energies_t = self.w(torch.tanh(Wh_t + U_t))

        ee = torch.cat((energies_v, energies_t), dim=1)  # [16, 2]
        weights_ee = torch.softmax(ee, dim=1)  # [16, 2]

        weights_v_t = weights_ee[:, None, :]

        values_v_Q = self.Q_v(values_v)
        values_t_Q = self.Q_t(values_t)


        # 加权—concat：
        # weights_v = weights_v_t[:, :, 0]
        # weights_t = weights_v_t[:, :, 1]
        #
        # attn_feat_v = weights_v.expand_as(values_v_Q)*values_v_Q
        # attn_feat_t = weights_t.expand_as(values_t_Q)*values_t_Q
        # attn_feats = torch.cat((attn_feat_v[:, None, :], attn_feat_t[:, None, :]), dim=2)  # [16, 1, 1200]


        # 加权求和：
        values_v_t = torch.cat((values_v_Q[:, None, :], values_t_Q[:, None, :]), dim=1)

        attn_feats = torch.bmm(weights_v_t, values_v_t)

        return attn_feats