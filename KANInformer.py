from kan import *
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, gau=False, device="cpu"):
        if gau:
            mask_shape = [B, L, L]
        else:
            mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv = KAN(width=[d_model*7, 20, d_model*7], grid=5, k=3)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = y.flatten(start_dim=1)
        y = self.conv(y)
        y = self.dropout(y.reshape(-1, 7, 64))
        print(y.shape)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv = KAN(width=[d_model*6, 20, d_model*6], grid=5, k=3)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = y.flatten(start_dim=1)
        y = self.conv(y)
        y = self.dropout(y.reshape(-1, 6, 64))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SpatialEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(SpatialEmbedding, self).__init__()
        self.spa_emb = nn.Embedding(c_in, d_model)

    def forward(self, x):
        x = x.long()
        spa_emb = self.spa_emb(x)

        return spa_emb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        self.freq = freq

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if self.freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, "10min": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


def exists(val):
    return val is not None


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)


# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = torch.einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


def series_to_supervised(data, n_in=2, n_out=1, dropna=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropna:
        agg.dropna(inplace=True)
    return agg


data = pd.read_csv(r"input.csv", header=None)
print(len(data))

N_step = 7
N_zong = data.shape[0]
N_heng = data.shape[1]

datas = np.zeros([N_heng, N_zong])
datass = np.zeros([N_heng, N_zong])

scaler = MinMaxScaler()

for i in range(0, N_heng):
    for j in range(0, N_zong):
        datas[i, j] = data.iloc[j][i]

for i in range(0, N_heng):
    datass[i][:, np.newaxis] = scaler.fit_transform(datas[i][:, np.newaxis])

ccc = [[0 for i in range(N_heng)] for j in range(N_zong)]
for i in range(0, N_heng):
    for j in range(0, N_zong):
        ccc[j][i] = datass[i, j]

df = pd.DataFrame(ccc)
df = df.astype('float32')
print(df)

reframed = series_to_supervised(df, N_step, 3)
print(reframed)

values = reframed.values
print(len(values))

train, val_test = train_test_split(values, test_size=0.2, shuffle=False)
val, test = train_test_split(val_test, test_size=0.5, shuffle=False)
print(len(train), len(val), len(test))

test_YY_1 = data.iloc[N_step + len(train) + len(val):N_step + len(train) + len(val) + len(test), -1]
test_YY_2 = data.iloc[N_step + len(train) + len(val) + 1:N_step + len(train) + len(val) + len(test) + 1, -1]
test_YY_3 = data.iloc[N_step + len(train) + len(val) + 2:N_step + len(train) + len(val) + len(test) + 2, -1]

train_X = train[:, :N_step * N_heng]
val_X = val[:, :N_step * N_heng]
test_X = test[:, :N_step * N_heng]

train_Y = train[:, -12:-8]
val_Y = val[:, -12:-8]

train_X = train_X.reshape(-1, N_step, N_heng)
val_X = val_X.reshape(-1, N_step, N_heng)
test_X = test_X.reshape(-1, N_step, N_heng)

train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
val_X = torch.from_numpy(val_X)
val_Y = torch.from_numpy(val_Y)
test_X = torch.from_numpy(test_X)

input_window = 7
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Informer(nn.Module):
    def __init__(self, args):
        super(Informer, self).__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        Attn = ProbAttention if args.enc_attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, args.factor, attention_dropout=args.dropout,
                             output_attention=args.output_attention),
                        args.d_model, args.n_heads, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads, mix=args.mix),
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
        )
        self.input_linear = nn.Linear(N_heng, args.d_model)
        self.output_linear = nn.Linear(args.d_model, N_heng)

    def forward(self, x_enc, x_mark_enc=None, x_mark_dec=None, enc_self_mask=None, dec_enc_mask=None):
        tgt = x_enc[:, 1:, :]
        src_start = self.input_linear(x_enc)
        src = self.enc_embedding(src_start, x_mark_enc)
        enc_out, _ = self.encoder(src, attn_mask=enc_self_mask)
        tgt_start = self.input_linear(tgt)
        tgt = self.dec_embedding(tgt_start, x_mark_dec)
        mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
        out = self.decoder(tgt, enc_out, mask, cross_mask=dec_enc_mask)
        out = self.output_linear(out)
        out = out[:, -1, :]
        return out

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


import argparse

parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic
parser.add_argument('--model', type=str, default='autoformer', help='model of experiment, options: [lstm, \
mlp, tpa, tcn, trans, gated, informerstack, informerlight(TBD)], autoformer, transformer,\
edlstm, edgru, edgruattention')
parser.add_argument('--data', type=str, default='Mydata',
                    help='only for revising some params related to the data, [ETTh1, Ubiquant]')
parser.add_argument('--dataset', type=str, default='Mydata', help='dataset, [ETTh1, Ubiquant]')
parser.add_argument('--data_path', type=str, default='./data/Mydata/', help='root path of the data file')
parser.add_argument('--file_name', type=str, default='Mydata.csv', help='file_name')
parser.add_argument('--criterion', type=str, default='mse', help='loss function')

# data
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# if features == "MS" or "S", need to provide target and target_pos
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--target_pos', type=int, default=-1, help='target feature position')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--seq_len', type=int, default=60, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=10, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')
parser.add_argument('--horizon', type=int, default=1,
                    help='predict timeseries horizon-th in head.When many2many, means from 1(default) to pred_len')
parser.add_argument('--inverse', action='store_true', help='inverse output data')
parser.add_argument('--scale', default=True, type=bool, help='scale input data')
parser.add_argument('--out_inverse', action='store_true', help='inverse output data')
parser.add_argument('--start_col', type=int, default=1, help='Index of the start column of the variables')
# training
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--itr', type=int, default=2, help='experiments times')

# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--des', type=str, default='test', help='exp description')

# model common
parser.add_argument('--out_size', type=int, default=7, help='output features size')
parser.add_argument('--dropout', type=float, default=0, help='dropout')

# seq2seq common
parser.add_argument('--enc_in', type=int, default=64, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=64, help='decoder input size')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='teacher_forcing_ratio')

# informer, autoformer, transformer
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--uv_size', type=int, default=2048, help='dimension of uv')
parser.add_argument('--qk_size', type=int, default=512, help='dimension of qk')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--enc_attn', type=str, default='prob',
                    help='attention used in encoder(only in informer), options:[gate, prob, full]')
parser.add_argument('--dec_selfattn', type=str, default='full',
                    help='selfattention used in encoder(in gau), options:[gate, prob, full]')
parser.add_argument('--dec_crossattn', type=str, default='full',
                    help='crossattention used in encoder(in gau), options:[gate, prob, full]')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--use_conv', action='store_true', help='use conv1d attention in GateAtten')
parser.add_argument('--use_bias', action='store_true', help='use bias in GateAtten')
parser.add_argument('--use_aff', action='store_true', help='use aff in GateAtten')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# nonseq2seq comon
parser.add_argument('--input_size', type=int, default=300, help='input features dim')
# LSTNet
parser.add_argument('--hidRNN', default=100, help='RNN hidden zie')
parser.add_argument('--hidCNN', type=int, default=100, help='CNN hidden size')
parser.add_argument('--hidSkip', type=float, default=5)
parser.add_argument('--skip', type=int, default=20)
parser.add_argument('--CNN_kernel', default=6, help='kernel size')
parser.add_argument('--highway_window', type=float, default=20, help='ar regression used last * items')

# tcn
parser.add_argument('--tcn_n_layers', default=3, help='num_layers')
parser.add_argument('--tcn_hidden_size', type=int, default=64, help='tcn hidden size')
parser.add_argument('--tcn_dropout', type=float, default=0.05, help='dropout')

# tpa
parser.add_argument('--tpa_n_layers', default=3, help='num_layers')
parser.add_argument('--tpa_hidden_size', type=int, default=64, help='tpa hidden size')
parser.add_argument('--tpa_ar_len', type=int, default=10, help='ar regression used last * items')

# trans
parser.add_argument('--trans_n_layers', default=3, help='num_layers')
parser.add_argument('--trans_hidden_size', type=int, default=256, help='trans hidden size')
parser.add_argument('--trans_n_heads', type=int, default=8, help='num of attention heads')
parser.add_argument('--trans_kernel_size', type=int, default=6, help='output size')

# lstm
parser.add_argument('--lstm_n_layers', type=int, default=2)
parser.add_argument('--lstm_hidden_size', type=int, default=64)

# mlp
parser.add_argument('--mlp_hidden_size', type=int, default=64)

# gated
parser.add_argument('--n_spatial', type=int, default=154, help='num of spatial')
parser.add_argument('--gdnn_embed_size', type=int, default=512, help='gdnn_embed_size')
parser.add_argument('--gdnn_hidden_size1', type=int, default=150, help='lstm hidden size')
parser.add_argument('--gdnn_hidden_size2', type=int, default=50, help=' combined model hidden size')
parser.add_argument('--gdnn_out_size', type=int, default=100, help='lstm output size')

# deepar
parser.add_argument('--data_folder', default='../timeseries-data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--load', default=True, help='load last trained model')

parser.add_argument('--val_num', type=int, default=4, help='val_num in one epoch')
parser.add_argument('--single_file', type=bool, default=True, help='single_file')
parser.add_argument('--debug', action='store_true', help='whether debug')
parser.add_argument('--input_params', type=str, nargs="+", default=["x", 'x_mark', 'y', 'y_mark'], help='input_params')
parser.add_argument('--target_param', type=str, default="y", help='target_params')
parser.add_argument('--test_year', type=int, default=2017, help='test year')
parser.add_argument('--importance', type=bool, default=False, help='importance')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data_path': './data/ETT/', 'file_name': 'ETTh1.csv', "dataset": "ETTh1",
              "freq": 'h',  # 'seq_len':96, 'label_len':48, "pred_len":24,
              "features": "M", 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1], "freq": 't'},
    'ETTm2': {'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    'Mydata': {'freq': 'b', 'T': 'rv', "features": "MS", 'MS': [22, 22, 1], 'M': [22, 22, 22]},
    "SDWPF": {'freq': '10min', 'T': 'Patv', "features": "MS", 'MS': [10, 10, 1], 'M': [10, 10, 10]},
    'Toy': {'data_path': './data/ToyData', 'seq_len': 96, 'label_len': 0, "pred_len": 24, "MS": [1, 1, 1], "T": "s"},
    'oze': {'seq_len': 672, 'label_len': 1, "pred_len": 671, "M": [37, 8, 8], "T": "s", 'features': "M"}
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.features = data_info.get("features") or args.features
    args.enc_in, args.dec_in, args.out_size = data_info[args.features]
    args.data_path = data_info.get("data_path") or args.data_path
    args.file_name = data_info.get("file_name") or args.file_name
    args.dataset = data_info.get("dataset") or args.dataset
    args.horizon = data_info.get("horizon") or args.horizon
    args.single_file = data_info.get("single_file") if data_info.get("single_file") is not None else args.single_file
    args.seq_len = data_info.get('seq_len') or args.seq_len
    args.label_len = data_info.get('label_len') or args.label_len
    args.pred_len = data_info.get('pred_len') or args.pred_len
    args.target = data_info['T']

    if 'freq' in data_info:
        args.freq = data_info["freq"]
args.input_size = args.enc_in
args.test_activation = "softmax"

args = parser.parse_args()

model = Informer(args).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def addbatch(data_train, data_test, batchsize):
    data = TensorDataset(data_train, data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=False)  # shuffle是是否打乱数据集，可自行设置
    return data_loader


traindata = addbatch(train_X, train_Y, 32)
valdata = addbatch(val_X, val_Y, 32)

best_val_loss = float('inf')
patience = 3
no_improvement_epochs = 0

for t in range(200):
    for step, (batch_x, batch_y) in enumerate(traindata):
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = 0.0
    with torch.no_grad():
        for val_step, (val_batch_x, val_batch_y) in enumerate(valdata):
            val_prediction = model(val_batch_x)
            val_loss += criterion(val_prediction, val_batch_y).item()
    val_loss /= (val_step + 1)

    print('Epoch: {}, Loss: {:.5f}, Val Loss: {:.5f}'.format(t + 1, loss.item(), val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= patience:
            print('Early stopping at epoch:', t + 1)
            model.load_state_dict(best_model_weights)
            break

predictions = []
for i in range(0, 3):
    pred_test = model(test_X).unsqueeze(1)
    test_X = torch.cat([test_X[:, 1:, :], pred_test], dim=1)
    predictions.append(pred_test.squeeze(1))

predictions_tensor = torch.stack(predictions, dim=1)

pred_test1 = predictions_tensor[:, 0, -1].view(-1).data.numpy()
pred_test2 = predictions_tensor[:, 1, -1].view(-1).data.numpy()
pred_test3 = predictions_tensor[:, 2, -1].view(-1).data.numpy()

origin_data1 = scaler.inverse_transform(pred_test1[:, np.newaxis])
origin_data2 = scaler.inverse_transform(pred_test2[:, np.newaxis])
origin_data3 = scaler.inverse_transform(pred_test3[:, np.newaxis])

c1 = np.zeros((len(origin_data1),), dtype=float)
c2 = np.zeros((len(origin_data2),), dtype=float)
c3 = np.zeros((len(origin_data3),), dtype=float)

for i in range(len(origin_data1)):
    c1[i] = origin_data1[i]
    c2[i] = origin_data2[i]
    c3[i] = origin_data3[i]


def rmse(records_real, record_predict):
    if len(records_real) == len(record_predict):
        return (sum([abs(x - y) ** 2 for x, y in zip(records_real, record_predict)]) / len(records_real)) ** 0.5


def mae(records_real, record_predict):
    if len(records_real) == len(record_predict):
        return sum([abs(x - y) for x, y in zip(records_real, record_predict)]) / len(records_real)


def mape(records_real, record_predict):
    if len(records_real) == len(record_predict):
        return 100 * sum([abs((x - y) / x) for x, y in zip(records_real, record_predict) if x != 0]) / len(records_real)


print("rmse:", rmse(test_YY_1, c1), rmse(test_YY_2, c2), rmse(test_YY_3, c3))
print("mae:", mae(test_YY_1, c1), mae(test_YY_2, c2), mae(test_YY_3, c3))
print("mape:", mape(test_YY_1, c1), mape(test_YY_2, c2), mape(test_YY_3, c3))
