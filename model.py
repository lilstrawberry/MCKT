import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import glo

def attention_score(query, key, value, mask, gamma):
    # batch head seq seq
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])

    seq = scores.shape[-1]
    x1 = torch.arange(seq).float().unsqueeze(-1).to(query.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask, -1e9)
        scores_ = torch.softmax(scores_, dim=-1)

        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :]  # 1 1 seq seq
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    gamma = -1.0 * gamma.abs().unsqueeze(0)  # 1 head 1 1
    total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

    scores = scores * total_effect
    scores = torch.masked_fill(scores, mask, -1e9)
    scores = torch.softmax(scores, dim=-1)
    scores = torch.masked_fill(scores, mask, 0)

    output = torch.matmul(scores, value)

    return output, scores

class MultiHead_Forget_Attn(nn.Module):
    def __init__(self, d, p, head):
        super(MultiHead_Forget_Attn, self).__init__()

        self.q_linear = nn.Linear(d, d)
        self.k_linear = nn.Linear(d, d)
        self.v_linear = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.head = head
        self.gammas = nn.Parameter(torch.zeros(head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, query, key, value, mask):
        # query: batch seq d
        batch = query.shape[0]
        origin_d = query.shape[-1]
        d_k = origin_d // self.head
        query = self.q_linear(query).view(batch, -1, self.head, d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch, -1, self.head, d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch, -1, self.head, d_k).transpose(1, 2)
        out, attn = attention_score(query, key, value, mask, self.gammas)
        # out, attn = getAttention(query, key, value, mask)
        # batch head seq d_k
        out = out.transpose(1, 2).contiguous().view(batch, -1, origin_d)
        out = self.linear_out(out)
        return out, attn

class TransformerLayer(nn.Module):
    def __init__(self, d, p, head):
        super(TransformerLayer, self).__init__()

        self.dropout = nn.Dropout(p)

        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.activation = nn.ReLU()
        self.attn = MultiHead_Forget_Attn(d, p, head)

    def forward(self, q, k, v, mask, apply=True):
        out, _ = self.attn(q, k, v, mask)
        q = q + self.dropout(out)
        q = self.layer_norm1(q)
        if apply:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
            q = q + self.dropout((query2))
            q = self.layer_norm2(q)
        return q

class MCKT(nn.Module):
    def __init__(self, pro_max, skill_max, d, p):
        super(MCKT, self).__init__()

        self.skill_max = skill_max

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))

        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))
        self.diff_embed = nn.Parameter(torch.rand(pro_max, 1))
        self.pro_change = nn.Parameter(torch.rand(skill_max, d))
        self.ans_embed = nn.Parameter(torch.rand(2, d))

        self.encoder = TransformerLayer(d, p, 8)
        self.decoder_1 = TransformerLayer(d, p, 8)
        self.decoder_2 = TransformerLayer(d, p, 8)

        self.lstm = nn.LSTM(d, d, batch_first=True)

        self.dropout = nn.Dropout(p=p)

        self.out = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1)
        )

    def sim(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        tau = 0.8
        res_sim = torch.matmul(a, b.transpose(-1, -2))
        c = res_sim / tau
        c = torch.exp(c)
        return c

    def batched_semi_loss(self, z1, batch_size):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / 0.8)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = self.sim(z1[mask], z1)  # [B, N]

            fenzi = (refl_sim * pos_matrix[mask]).sum(dim=-1)
            fenmu = refl_sim.sum(dim=-1)

            losses.append(-torch.log(fenzi / fenmu))

        return torch.cat(losses).mean()

    def state_cl(self, h_state, T, next_mask):

        batch = h_state.shape[0]
        seq = h_state.shape[1]
        d = h_state.shape[-1]

        temp_h = h_state.transpose(0, 1)  # seq batch d
        neg_sim = self.sim(temp_h, temp_h)  # seq batch batch

        pos_sim = self.sim(h_state, h_state)  # batch seq seq
        pos_use = torch.zeros((batch, seq, T)).to(h_state.device)  # batch seq T

        for i in range(T, seq):
            pos_use[:, i] = pos_sim[:, i, i - T: i]

        pos_use_matrix = pos_use.transpose(0, 1)  # seq batch T
        neg_use_matrix = neg_sim  # seq batch batch

        loss = []

        batch_seq = next_mask.shape[-1]
        batch_min_seq = next_mask.sum(dim=-1).min()

        for i in range(batch_seq - batch_min_seq + T, batch_seq):
            item_pos = pos_use_matrix[i]  # batch T
            item_neg = neg_use_matrix[i]  # batch batch

            fenmu = torch.sum(item_pos, dim=-1) + torch.sum(item_neg, dim=-1) - torch.diag(item_neg)
            fenzi = torch.sum(item_pos, dim=-1)

            item_loss = -torch.log(fenzi / fenmu)
            loss.append(item_loss)

        if len(loss) > 0:
            return torch.vstack(loss).mean()
        else:
            return 0

    def pro_similar(self, pro_embed):
        a = F.normalize(pro_embed, dim=-1)
        res = torch.matmul(a, a.transpose(-1, -2))
        return res

    def contrast_state_cl(self, h_state, other_state, next_mask):
        cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        batch = h_state.shape[0]
        seq = h_state.shape[1]
        d = h_state.shape[-1]

        temp_h = h_state.transpose(0, 1)  # seq batch d
        neg_sim = self.sim(temp_h, other_state.transpose(0, 1))  # seq batch batch

        loss = []

        batch_seq = next_mask.shape[-1]
        batch_min_seq = next_mask.sum(dim=-1).min()

        h_labels = torch.arange(h_state.size(0)).long().to(h_state.device)

        for i in range(batch_seq - batch_min_seq, batch_seq):
            item_pos = neg_sim[i]  # batch batch
            loss.append(cl_loss_fn(item_pos, h_labels))

        if len(loss) > 0:
            return torch.vstack(loss).mean()
        else:
            return 0

    def batched_semi_loss(self, z1, batch_size):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        tau = 0.8
        f = lambda x: torch.exp(x / tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        pos_matrix = glo.get_value('pos_matrix')

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]

            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            now_use_matrix = pos_matrix[mask]  # [B, N]

            fenzi = (refl_sim * now_use_matrix).sum(dim=-1, keepdims=True)  # B
            fenmu = refl_sim.sum(dim=-1, keepdims=True)  # B

            losses.append(-torch.log(fenzi / fenmu))

        return torch.cat(losses).mean()

    def forward(self, last_problem, last_ans, next_problem, next_ans):

        next_mask = next_ans != -1

        last_ans[last_ans < 0] = 0
        next_ans[next_ans < 0] = 0

        device = last_problem.device
        seq = last_problem.shape[-1]

        pro_embed = self.pro_embed

        pro_loss = glo.get_value('pro_cl') * self.batched_semi_loss(pro_embed, glo.get_value('cl_use_batch'))

        react_loss = glo.get_value('inter_cl') * self.batched_semi_loss(pro_embed + self.ans_embed[1].unsqueeze(0), glo.get_value('cl_use_batch'))

        next_pro_embed = F.embedding(next_problem, pro_embed)
        last_pro_embed = F.embedding(last_problem, pro_embed)

        next_X = next_pro_embed + F.embedding(next_ans.long(), self.ans_embed)
        last_X = last_pro_embed + F.embedding(last_ans.long(), self.ans_embed)

        pro_sim = self.pro_similar(next_pro_embed)  # batch seq seq
        T_yuzhi = glo.get_value('sim')

        pro_mask = (pro_sim < T_yuzhi).unsqueeze(1)

        mask = (torch.triu(torch.ones((seq, seq)), 1) == 1).to(device).unsqueeze(0).unsqueeze(0) | pro_mask
        l_mask = (torch.triu(torch.ones((seq, seq)), 0) == 1).to(device).unsqueeze(0).unsqueeze(0) | pro_mask

        encoder_out = self.encoder(next_pro_embed, next_pro_embed, next_pro_embed, mask)
        f2 = self.decoder_1(next_X, next_X, next_X, mask, False)
        decoder_out = self.decoder_2(encoder_out, f2, f2, l_mask)

        next_state, _ = self.lstm(self.dropout(last_X))

        now_use = torch.cat([decoder_out, next_state, next_pro_embed], dim=-1)

        P = torch.sigmoid(self.out(self.dropout(now_use))).squeeze(-1)

        state_loss = self.contrast_state_cl(next_state, decoder_out, next_mask) + self.contrast_state_cl(decoder_out,
                                                                                                         next_state,
                                                                                                         next_mask)
        state_loss = state_loss * glo.get_value('state_cl')

        return P, state_loss, pro_loss, react_loss