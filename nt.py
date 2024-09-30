import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, scatter, to_dense_batch
from performer_pytorch import SelfAttention


sigmoid = lambda x, *args: torch.sigmoid(x)


class NTLayer(nn.Module):
    def __init__(
        self, din, dout, heads=1, edge_dim=0, dropout=0,
        frame='gt', agg='weightedmean', **kwargs
    ):
        super(self.__class__, self).__init__()
        self.e = None
        self.e_raw = None
        self.order = None
        self.heads = H = heads
        if agg in ('weightedmean', 'gatedsum'):
            dout *= 2
        C = dout // H
        self.agg = agg
        self.comb = nn.Sequential(
            nn.Linear(din * 2 + edge_dim, dout), nn.GELU())
        if frame == 'performer':
            self.attn = SelfAttention(
                dim=dout, heads=H, dim_head=C, qkv_bias=True,
                dropout=dropout, causal=False)
            self.attn_fwd = lambda x, m: self.attn(x, mask=m)[m]
        else:
            self.attn = nn.MultiheadAttention(
                dout, H, dropout=dropout, batch_first=True)
            self.attn_fwd = lambda x, m: self.attn(
                x, x, x, key_padding_mask=~m, need_weights=False)[0][m]

    def forward(self, x, e, edge_attr=None, batch=None, **kwargs):
        if self.e_raw is not e:
            self.e_raw = e
            self.order = e[0].sort().indices
            self.e = e[:, self.order]
        e = self.e
        n, m = x.shape[0], e.shape[1]
        ex = list(x[e])
        if edge_attr is not None:
            ex.append(edge_attr[self.order])
        ex = F.gelu(self.comb(torch.cat(ex, dim=1)))
        # NOTE: e[0] must be ordered and consecutive.
        # It is not guaranteed to be consecutive here
        # because NTLayer is just experimental.
        h = F.gelu(self.attn_fwd(*to_dense_batch(ex, e[0])))
        agg = {'weightedmean': softmax, 'gatedsum': sigmoid}.get(self.agg)
        if agg:
            h = h.view(m, self.heads, 2, -1)
            alpha = agg(h[:, :, 1].mean(dim=-1), e[0], None, n)
            h = scatter(
                h[:, :, 0] * alpha.unsqueeze(-1),
                e[1], dim=0, dim_size=n, reduce='sum'
            ).view(n, -1)
        else:
            h = scatter(h, e[1], dim=0, dim_size=n, reduce=self.agg)
        return h


def divide_edges(e0, pfsep, factor=0.4):
    _, inv1, ds = e0.unique(return_inverse=True, return_counts=True)
    ds, inv2, cs = ds.unique(
        sorted=True, return_inverse=True, return_counts=True)
    acc = cs.cumsum(dim=0)
    area = acc[-1] * ds[-1]
    print('#nodes: %d, max degree: %d, space: %d'
          % (acc[-1], ds[-1], area))
    todo = []
    # Divide neighbourhoods by size
    use_pf, k = (ds > pfsep).long().max(dim=0)
    if k.item() == 0:
        todo.append((area, 0, ds.shape[0], bool(use_pf)))
    else:
        area_l = acc[k-1] * ds[k-1]
        area_r = area - acc[k-1] * ds[-1]
        todo.extend([(area_l, 0, k, False), (area_r, k, ds.shape[0], True)])
    # Divide neighbourhoods by area
    stop, done = False, []
    while todo:
        todo.sort()
        area, i, j, use_pf = todo.pop()
        _ds, _acc = ds[i:j], acc[i:j] - (i and acc[i-1])
        maxd, n = _ds[-1], _acc[-1]
        if not stop and j - i > 1:
            area_l, area_r = _acc * _ds, area - _acc * maxd
            area_m, k = torch.max(area_l, area_r).min(dim=0)
            if area_m.item() < factor * area:
                area_l = area_l[k].item()
                area_r = area_r[k].item()
                k = k.item() + 1 + i
                todo.extend([(area_l, i, k, use_pf), (area_r, k, j, use_pf)])
                continue
        stop = True
        sec = ((inv2 >= i) & (inv2 < j))[inv1]
        ids = e0[sec].unique(return_inverse=True)[1]
        done.append([sec, ids, maxd, use_pf])
        print('Section: %d >= deg >= %d, #nodes: %d, space: %d, use_pf: %s'
              % (maxd, _ds[0], n, area, use_pf))
    return done


e_raws = []
e_processes = []
e_pairs = []


class NTPLayer(nn.Module):
    def __init__(
        self, din, dout, heads=1, edge_dim=0, dropout=0,
        frame='mix', agg='weightedmean', divide_factor=0.4,
        pf_threshold=-1, **kwargs
    ):
        super(self.__class__, self).__init__()
        self.heads = heads
        if agg in ('weightedmean', 'gatedsum'):
            dout *= 2
        self.dim_head = dout // heads
        self.agg = agg
        self.divide_factor = divide_factor
        self.comb = nn.Linear(din * 2 + edge_dim, dout)
        self.attn = SelfAttention(
            dim=dout, heads=heads, dim_head=self.dim_head,
            qkv_bias=True, dropout=dropout, causal=False)
        self.pf_threshold = pf_threshold
        if pf_threshold < 0:
            pfm = self.attn.fast_attention.nb_features
            self.pf_threshold = (pfm * (pfm + self.dim_head)) ** 0.5 + pfm
        print('Performer threshold:', self.pf_threshold)
        self.pf_fwd = lambda x, m: self.attn(x, mask=m)[m]
        self.gt_fwd = lambda x, m: self.attn.dropout(
            F.multi_head_attention_forward(
                *([x.transpose(1, 0)] * 3), dout, heads,
                in_proj_weight=torch.cat((
                    self.attn.to_q.weight,
                    self.attn.to_k.weight,
                    self.attn.to_v.weight)),
                in_proj_bias=torch.cat((
                    self.attn.to_q.bias,
                    self.attn.to_k.bias,
                    self.attn.to_v.bias)),
                bias_k=None, bias_v=None,
                add_zero_attn=False, dropout_p=0,
                out_proj_weight=self.attn.to_out.weight,
                out_proj_bias=self.attn.to_out.bias,
                training=self.training, key_padding_mask=~m,
                need_weights=False, attn_mask=None,
                use_separate_proj_weight=False,
                q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
                static_k=None, static_v=None,
                average_attn_weights=True, is_causal=False,
            )[0].transpose(1, 0)
        )[m]

    def forward(self, x, e, edge_attr=None, batch=None, **kwargs):
        for e_raw, e_process in zip(e_raws, e_processes):
            if e_raw is e:
                e, order, secs = e_process
                break
        else:
            e_raws.append(e)
            # NOTE: MUST be sorted
            e0, order = e[0].sort()
            e = e[:, order]
            secs = divide_edges(e0, self.pf_threshold, self.divide_factor)
            e_processes.append((e, order, secs))
        n, m = x.shape[0], e.shape[1]
        ex = list(x[e])
        if edge_attr is not None:
            ex.append(edge_attr[order])
        ex = F.gelu(self.comb(torch.cat(ex, dim=1)))
        h = x.new_zeros(ex.shape)
        for sec, ids, maxd, use_pf in secs:
            h[sec] = (self.pf_fwd if use_pf else self.gt_fwd)(
                *to_dense_batch(ex[sec], ids))
        h = F.gelu(h)
        agg = {'weightedmean': softmax, 'gatedsum': sigmoid}.get(self.agg)
        if agg:
            h = h.view(m, self.heads, 2, -1)
            alpha = agg(h[:, :, 1].mean(dim=-1), e[0], None, n)
            h = scatter(
                h[:, :, 0] * alpha.unsqueeze(-1),
                e[1], dim=0, dim_size=n, reduce='sum'
            ).view(n, -1)
        else:
            h = scatter(h, e[1], dim=0, dim_size=n, reduce=self.agg)
        return h


class NT(nn.Module):
    def __init__(
            self, in_channels, out_channels, hidden, n_layers,
            heads, dropout, is_bidir=True, edge_dim=0, frame='mix',
            sep=False, **kwargs):
        super(self.__class__, self).__init__()
        self.sep = sep
        hidden_channels = hidden * heads
        self.enc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout),
            nn.GELU())
        self.e_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.Dropout(dropout),
            nn.GELU()) if edge_dim else nn.Identity()
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.convs = nn.ModuleList([
            {'gt': NTLayer, 'performer': NTLayer}.get(frame, NTPLayer)(
                hidden_channels, hidden_channels, heads,
                edge_dim=edge_dim and hidden_channels,
                dropout=dropout, **kwargs)
            for _ in range(n_layers)])
        self.is_bidir = is_bidir
        if not is_bidir:
            self.rconvs = nn.ModuleList([
                {'gt': NTLayer, 'performer': NTLayer}.get(frame, NTPLayer)(
                    hidden_channels, hidden_channels, heads,
                    edge_dim=edge_dim and hidden_channels,
                    dropout=dropout, **kwargs)
                for _ in range(n_layers)])
        self.fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * (1 + sep), hidden_channels),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout))
            for _ in range(n_layers)])
        self.pred = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels))

    def forward(self, x, e, edge_attr=None, **kwargs):
        x = self.enc(x)
        if edge_attr is not None:
            edge_attr = self.e_enc(edge_attr)
        if self.is_bidir:
            for norm, conv, fn in zip(self.norms, self.convs, self.fns):
                nx = norm(x)
                h = conv(nx, e, edge_attr)
                if self.sep:
                    h = torch.cat((nx, h), dim=-1)
                x = x + fn(h)
        else:
            for _e, re in e_pairs:
                if _e is e:
                    break
            else:
                re = e[[1, 0]]
                e_pairs.extend([(e, re), (re, e)])
            for norm, conv, rconv, fn in zip(
                    self.norms, self.convs, self.rconvs, self.fns):
                nx = norm(x)
                h = conv(nx, e, edge_attr) + rconv(nx, re, edge_attr)
                if self.sep:
                    h = torch.cat((nx, h), dim=-1)
                x = x + fn(h)
        x = self.pred(x)
        return x
