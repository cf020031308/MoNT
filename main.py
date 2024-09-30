import os
import time
import json
import random
import datetime
import argparse
import psutil

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric import datasets as pyg_data
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import f1_score, roc_auc_score
import torch_sparse
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected


parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, default='MLP', help=(
    'MLP | SGC | LPA | PPNP | PPNR | GIN | GCN | SAGE | GCNII'
))
parser.add_argument('dataset', type=str, default='cora', help=(
    'cora | citeseer | pubmed | flickr | arxiv | yelp | reddit | ...'
))
parser.add_argument('--runs', type=int, default=1, help='Default: 1')
parser.add_argument('--gpu', type=int, default=0, help='Default: 0')
parser.add_argument(
    '--split', type=float, default=0,
    help=('Ratio of labels for training.'
          ' Set to 0 to use default split (if any) or 0.6. '
          ' With an integer x the dataset is splitted like Cora with the '
          ' training set be composed by x samples per class. '
          ' Default: 0'))
parser.add_argument(
    '--lr', type=float, default=0.001, help='Learning Rate. Default: 0.001')
parser.add_argument(
    '--dropout', type=float, default=0.0, help='Default: 0')
parser.add_argument('--n-layers', type=int, default=2, help='Default: 2')
parser.add_argument(
    '--weight-decay', type=float, default=0.0, help='Default: 0')
parser.add_argument(
    '--early-stop-epochs', type=int, default=200,
    help='Maximum epochs until stop when accuracy decreasing. Default: 200')
parser.add_argument(
    '--max-epochs', type=int, default=1000,
    help='Maximum epochs. Default: 1000')
parser.add_argument(
    '--hidden', type=int, default=32,
    help='Dimension of hidden representations and implicit state. Default: 32')
parser.add_argument(
    '--heads', type=int, default=1,
    help='Heads for GAT. Default: 1')
parser.add_argument(
    '--alpha', type=float, default=0.1,
    help='Hyperparameter for GCNII. Default: 0.1')
parser.add_argument(
    '--beta', type=float, default=0.0,
    help='Hyperparameter for GCNII. Default: 0.05')
parser.add_argument(
    '--add-self-loops', action='store_true',
    help='add self-loops')
parser.add_argument(
    '--remove-self-loops', action='store_true',
    help='remove self-loops')
parser.add_argument(
    '--to-bidir', action='store_true',
    help='to bidirectional graph if it is not')
parser.add_argument(
    '--sep', action='store_true',
    help='separate ego- and neighbour-features')
parser.add_argument(
    '--frame', type=str, default='mix',
    help='version of implementation')
parser.add_argument(
    '--agg', type=str, default='weightedmean',
    help='aggregation method for Message Sharing')
parser.add_argument(
    '--divide-factor', type=float, default=0.4,
    help='Divide factor for Message Sharing')
parser.add_argument(
    '--pf-threshold', type=float, default=-1,
    help='Neighbourhood scale threshold to enable performer. Default: -1 (Auto)')
args = parser.parse_args()

inf = float('inf')
if not torch.cuda.is_available():
    args.gpu = -1
print(datetime.datetime.now(), args)
script_time = time.time()

g_dev = None
gpu = lambda x: x
if args.gpu >= 0:
    g_dev = torch.device('cuda:%d' % args.gpu)
    gpu = lambda x: x.to(g_dev)
coo = torch.sparse_coo_tensor


def fix_seed(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Optim(object):
    def __init__(self, params):
        self.params = params
        self.opt = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)

    def __repr__(self):
        return 'params: %d' % sum(p.numel() for p in self.params)

    def __enter__(self):
        self.opt.zero_grad()
        self.elapsed = time.time()
        return self.opt

    def __exit__(self, *vs, **kvs):
        self.opt.step()
        self.elapsed = time.time() - self.elapsed


class GCNII(nn.Module):
    def __init__(self, din, hidden, n_layers, dout, dropout=0, **kw):
        super(self.__class__, self).__init__()
        self.lin1 = nn.Linear(din, hidden)
        self.convs = nn.ModuleList([
            gnn.GCN2Conv(
                channels=hidden,
                alpha=kw['alpha'],
                theta=kw['beta'],
                layer=i + 1,
            ) for i in range(n_layers)])
        self.lin2 = nn.Linear(hidden, dout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x0 = x = F.relu(self.lin1(self.dropout(x)))
        for conv in self.convs:
            x = F.relu(conv(self.dropout(x), x0, edge_index))
        return self.lin2(self.dropout(x))


def load_data(name):
    is_bidir = None
    train_masks = None
    W = None
    if args.dataset in (
        'roman_empire', 'amazon_ratings', 'minesweeper',
        'tolokers', 'questions',
    ):
        data = numpy.load('dataset/hetgs/%s.npz' % args.dataset)
        X, Y, E, train_masks, valid_masks, test_masks = map(data.get, [
            'node_features', 'node_labels', 'edges',
            'train_masks', 'val_masks', 'test_masks'])
        X, Y, E, train_masks, valid_masks, test_masks = map(torch.from_numpy, [
            X, Y, E.T, train_masks, valid_masks, test_masks])
        is_bidir = False
    elif args.dataset in ('arxiv', 'mag', 'products', 'proteins'):
        ds = NodePropPredDataset(name='ogbn-%s' % args.dataset)
        train_idx, valid_idx, test_idx = map(
            ds.get_idx_split().get, 'train valid test'.split())
        if args.dataset == 'mag':
            train_idx = train_idx['paper']
            valid_idx = valid_idx['paper']
            test_idx = test_idx['paper']
        g, labels = ds[0]
        if args.dataset == 'mag':
            labels = labels['paper']
            g['edge_index'] = g['edge_index_dict'][('paper', 'cites', 'paper')]
            g['node_feat'] = g['node_feat_dict']['paper']
        E = torch.from_numpy(g['edge_index'])
        if args.dataset == 'proteins':
            W = torch.from_numpy(g['edge_feat'])
            X = torch.zeros(g['num_nodes'], W.shape[1]).scatter_add_(
                dim=0,index=E[0].view(-1, 1).expand_as(W), src=W)
        else:
            W = None
            X = torch.from_numpy(g['node_feat'])
        Y = torch.from_numpy(labels).squeeze(-1)
        n_nodes = X.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True
        is_bidir = False
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    else:
        dn = 'dataset/' + args.dataset
        g = (
            pyg_data.Planetoid(dn, name='Cora') if args.dataset == 'cora'
            else pyg_data.Planetoid(dn, name='CiteSeer') if args.dataset == 'citeseer'
            else pyg_data.Planetoid(dn, name='PubMed') if args.dataset == 'pubmed'
            else pyg_data.CitationFull(dn, name='Cora') if args.dataset == 'corafull'
            else pyg_data.CitationFull(dn, name='Cora_ML') if args.dataset == 'coraml'
            else pyg_data.CitationFull(dn, name='DBLP') if args.dataset == 'dblp'
            else pyg_data.Reddit(dn) if args.dataset == 'reddit'
            else pyg_data.Reddit2(dn) if args.dataset == 'reddit-sp'
            else pyg_data.Flickr(dn) if args.dataset == 'flickr'
            else pyg_data.Yelp(dn) if args.dataset == 'yelp'
            else pyg_data.AmazonProducts(dn) if args.dataset == 'amazon'
            else pyg_data.WebKB(dn, args.dataset.capitalize())
            if args.dataset in ('cornell', 'texas', 'wisconsin')
            else pyg_data.WikipediaNetwork(dn, args.dataset)
            if args.dataset in ('chameleon', 'crocodile', 'squirrel')
            else pyg_data.WikiCS(dn) if args.dataset == 'wikics'
            else pyg_data.Actor(dn) if args.dataset == 'actor'
            else pyg_data.Coauthor(dn, name='CS') if args.dataset == 'coauthor-cs'
            else pyg_data.Coauthor(dn, name='Physics') if args.dataset == 'coauthor-phy'
            else pyg_data.Amazon(dn, name='Computers') if args.dataset == 'amazon-com'
            else pyg_data.Amazon(dn, name='Photo') if args.dataset == 'amazon-photo'
            else None
        ).data
        X, Y, E, train_mask, valid_mask, test_mask = map(
            g.get, 'x y edge_index train_mask val_mask test_mask'.split())
        if args.dataset in (
                'amazon-com', 'amazon-photo', 'coauthor-cs', 'coauthor-phy'):
            train_mask, valid_mask, test_mask = torch.zeros(
                (3, X.shape[0]), dtype=bool)
            train_idx, valid_idx, test_idx = map(
                numpy.load('dataset/split/%s.npz' % args.dataset).get,
                ['train', 'valid', 'test'])
            train_mask[train_idx] = True
            valid_mask[valid_idx] = True
            test_mask[test_idx] = True
        elif args.dataset in (
            'cora', 'citeseer', 'pubmed', 'corafull',
            'reddit', 'flickr', 'yelp', 'amazon'
        ):
            train_masks = [train_mask] * args.runs
            valid_masks = [valid_mask] * args.runs
            test_masks = [test_mask] * args.runs
            is_bidir = True
        elif args.dataset in ('wikics', ):
            train_masks = train_mask.T
            valid_masks = valid_mask.T
            test_masks = [test_mask] * args.runs
        else:
            train_masks = [train_mask[:, i % train_mask.shape[1]]
                           for i in range(args.runs)]
            valid_masks = [valid_mask[:, i % valid_mask.shape[1]]
                           for i in range(args.runs)]
            test_masks = [test_mask[:, i % test_mask.shape[1]]
                          for i in range(args.runs)]
            is_bidir = False
    if is_bidir is None:
        for i in range(E.shape[1]):
            src, dst = E[:, i]
            if src.item() != dst.item():
                print(src, dst)
                break
        is_bidir = ((E[0] == dst) & (E[1] == src)).any().item()
        print('guess is bidir:', is_bidir)
    n_labels = int(Y.max().item() + 1)
    is_multilabel = len(Y.shape) == 2
    # Save Label Transitional Matrices
    fn = 'dataset/labeltrans/%s.json' % args.dataset
    if not (is_multilabel or os.path.exists(fn)):
        with open(fn, 'w') as file:
            mesh = coo(
                Y[E], torch.ones(E.shape[1]), size=(n_labels, n_labels)
            ).to_dense()
            den = mesh.sum(dim=1, keepdim=True)
            mesh /= den
            mesh[den.squeeze(1) == 0] = 0
            json.dump(mesh.tolist(), file)
    if (train_masks is None or train_masks[0] is None) and not args.split:
        args.split = 0.6
    nrange = torch.arange(X.shape[0])
    if 0 < args.split < 1:
        fix_seed(42)
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            valid_mask = torch.zeros(X.shape[0], dtype=bool)
            test_mask = torch.zeros(X.shape[0], dtype=bool)
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
            if is_multilabel:
                val_num = test_num = int((1 - args.split) / 2 * X.shape[0])
                idx = torch.randperm(X.shape[0])
                train_mask[idx[val_num + test_num:]] = True
                valid_mask[idx[:val_num]] = True
                test_mask[idx[val_num:val_num + test_num]] = True
            else:
                for c in range(n_labels):
                    label_idx = nrange[Y == c]
                    val_num = test_num = int(
                        (1 - args.split) / 2 * label_idx.shape[0])
                    perm = label_idx[torch.randperm(label_idx.shape[0])]
                    train_mask[perm[val_num + test_num:]] = True
                    valid_mask[perm[:val_num]] = True
                    test_mask[perm[val_num:val_num + test_num]] = True
    elif int(args.split):
        # NOTE: work only for graphs with single labelled nodes.
        fix_seed(42)
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            for y in range(n_labels):
                label_mask = Y == y
                train_mask[
                    nrange[label_mask][
                        torch.randperm(label_mask.sum())[:int(args.split)]]
                ] = True
            valid_mask = ~train_mask
            valid_mask[
                nrange[valid_mask][torch.randperm(valid_mask.sum())[500:]]
            ] = False
            test_mask = ~(train_mask | valid_mask)
            test_mask[
                nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]
            ] = False
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
    return X, Y, E, W, list(train_masks), list(valid_masks), list(test_masks), is_bidir


class Stat(object):
    def __init__(self):
        self.preprocess_time = 0
        self.training_times = []
        self.evaluation_times = []

        self.best_test_scores = []
        self.best_times = []
        self.best_training_times = []

        self.mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            self.gpu = torch.cuda.memory_allocated(g_dev) / 1024 / 1024

    def start_preprocessing(self):
        self.preprocess_time = time.time()

    def stop_preprocessing(self):
        self.preprocess_time = time.time() - self.preprocess_time

    def start_run(self):
        self.params = None
        self.scores = []
        self.acc_training_times = []
        self.acc_times = []
        self.training_times.append(0.)
        self.evaluation_times.append(0.)

    def record_training(self, elapsed):
        self.training_times[-1] += elapsed

    def record_evaluation(self, elapsed):
        self.evaluation_times[-1] += elapsed

    def evaluate_result(self, y):
        self.scores.append([
            get_score(Y[m], y[m])
            for m in [train_mask, valid_mask, test_mask]])
        self.acc_training_times.append(self.training_times[-1])
        self.acc_times.append(self.preprocess_time + self.training_times[-1])
        self.best_epoch = torch.tensor(self.scores).max(dim=0).indices[1] + 1
        dec_epochs = len(self.scores) - self.best_epoch
        if dec_epochs == 0:
            self.best_acc = self.scores[-1][1]
            self.best_y = y
        return dec_epochs >= args.early_stop_epochs

    def end_run(self):
        if self.scores:
            self.scores = torch.tensor(self.scores)
            print('train scores:', self.scores[:, 0].tolist())
            print('val scores:', self.scores[:, 1].tolist())
            print('test scores:', self.scores[:, 2].tolist())
            print('acc training times:', self.acc_training_times)
            print('max scores:', self.scores.max(dim=0).values)
            idx = self.scores.max(dim=0).indices[1]
            self.best_test_scores.append((idx, self.scores[idx, 2]))
            self.best_training_times.append(self.acc_training_times[idx])
            self.best_times.append(self.acc_times[idx])
            print('best test score:', self.best_test_scores[-1])

    def end_all(self):
        conv = 1.0 + torch.tensor([
            idx for idx, _ in self.best_test_scores])
        score = 100 * torch.tensor([
            score for _, score in self.best_test_scores])
        tm = torch.tensor(self.best_times)
        ttm = torch.tensor(self.best_training_times)
        print('converge time: %.3f±%.3f' % (
            tm.mean().item(), tm.std().item()))
        print('converge training time: %.3f±%.3f' % (
            ttm.mean().item(), ttm.std().item()))
        print('converge epochs: %.3f±%.3f' % (
            conv.mean().item(), conv.std().item()))
        print('score: %.2f±%.2f' % (score.mean().item(), score.std().item()))

        # Output Used Time
        print('preprocessing time: %.3f' % self.preprocess_time)
        for name, times in (
            ('total training', self.training_times),
            ('total evaluation', self.evaluation_times),
        ):
            times = torch.tensor(times or [0], dtype=float)
            print('%s time: %.3f±%.3f' % (
                name, times.mean().item(), times.std().item()))

        # Output Used Space
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu = torch.cuda.max_memory_allocated(g_dev) / 1024 / 1024
        print('pre_memory: %.2fM + %.2fM = %.2fM' % (
            self.mem, self.gpu, self.mem + self.gpu))
        print('max_memory: %.2fM + %.2fM = %.2fM' % (
            mem, gpu, mem + gpu))
        print('memory_diff: %.2fM + %.2fM = %.2fM' % (
            mem - self.mem,
            gpu - self.gpu,
            mem + gpu - self.mem - self.gpu))


X, Y, E, W, train_masks, valid_masks, test_masks, is_bidir = load_data(args.dataset)
n_nodes = X.shape[0]
n_features = X.shape[1]
is_multilabel = len(Y.shape) == 2
n_labels = Y.shape[1] if is_multilabel else int(Y.max().item() + 1)
deg = E.shape[1] / n_nodes
print('nodes: %d' % n_nodes)
print('features: %d' % n_features)
print('classes: %d' % n_labels)
print('is_multilabel:', is_multilabel)
print('is_bidir:', is_bidir)
print('edges: %d' % E.shape[1])
print('average degree: %.2f' % deg)
train_sum = sum([m.sum() for m in train_masks]) / len(train_masks)
valid_sum = sum([m.sum() for m in valid_masks]) / len(valid_masks)
test_sum = sum([m.sum() for m in test_masks]) / len(test_masks)
print('split: %d (%.2f%%) / %d (%.2f%%) / %d (%.2f%%)' % (
    train_sum, 100 * train_sum / n_nodes,
    valid_sum, 100 * valid_sum / n_nodes,
    test_sum, 100 * test_sum / n_nodes,
))
eh = (
    (Y[E[0]] == Y[E[1]]).sum().float()
    / E.shape[1] / (n_labels if is_multilabel else 1))
print('intra_rate: %.2f%%' % (100 * eh))

ds = torch.zeros(n_nodes)
ds.scatter_add_(dim=0, index=E[0], src=torch.ones(E.shape[1]))
if is_multilabel:
    ds = ds.unsqueeze(-1).repeat(1, n_labels)
    hs = torch.zeros(n_nodes, n_labels)
    for i in range(n_labels):
        hs[:, i].scatter_add_(
            dim=0, index=E[0], src=(Y[E[0], i] == Y[E[1], i]).float())
else:
    hs = torch.zeros(n_nodes)
    hs.scatter_add_(dim=0, index=E[0], src=(Y[E[0]] == Y[E[1]]).float())
nh = (hs / ds)[ds > 0].mean()
print('node homophily: %.2f%%' % (100 * nh))

if not is_multilabel:
    d2 = sum([ds[Y == i].sum() ** 2 for i in range(n_labels)])
    d2 *= E.shape[1] ** -2
    ah = (eh - d2) / (1 - d2)
    print('adjusted homophily: %.2f%%' % (100 * ah))

if is_multilabel:
    _cri = nn.BCEWithLogitsLoss(reduction='none')
    criterion = lambda x, y: _cri(x, y.float()).sum(dim=1)
    sg = torch.sigmoid
    if args.dataset in ('proteins', ):
        get_score = lambda y_true, y_pred: roc_auc_score(
            y_true.cpu(), y_pred.cpu()).item()
    else:
        get_score = lambda y_true, y_pred: f1_score(
            y_true.cpu(), (y_pred > 0.5).cpu(), average='micro').item()
else:
    criterion = lambda x, y: F.cross_entropy(x, y, reduction='none')
    sg = lambda x: torch.softmax(x, dim=-1)
    if args.dataset in ('minesweeper', 'tolokers', 'questions', ):
        get_score = lambda y_true, y_pred: roc_auc_score(
            y_true.cpu(), (1 - y_pred[:, 0]).cpu()).item()
    else:
        get_score = lambda y_true, y_pred: f1_score(
            y_true.cpu(), y_pred.argmax(dim=-1).cpu(), average='micro').item()


if not is_bidir and args.to_bidir:
    E = to_undirected(E)
    is_bidir = True
if args.remove_self_loops or args.add_self_loops:
    E, _ = remove_self_loops(E)
if args.add_self_loops:
    E, _ = add_self_loops(E)
ev = Stat()
opt = None

# Preprocessing
ev.start_preprocessing()

X, Y = map(gpu, [X, Y])
E = gpu(E)
A = None

ev.stop_preprocessing()

for run in range(args.runs):
    train_mask = train_masks[run]
    valid_mask = valid_masks[run]
    test_mask = test_masks[run]
    if is_multilabel:
        train_y = Y[train_mask].float()
    else:
        train_y = F.one_hot(Y[train_mask], n_labels).float()
        
    fix_seed(run)
    ev.start_run()

    if args.method == 'MLP':
        net = gpu(gnn.MLP(
            [n_features, *([args.hidden] * (args.n_layers - 1)), n_labels],
            dropout=args.dropout))
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                z = net(X)
                criterion(z[train_mask], Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)
            if ev.evaluate_result(sg(z)):
                break
    else:
        if args.method == 'NT':
            from nt import NT
            net = NT(n_features, n_labels, is_bidir=is_bidir, **args.__dict__)
        elif args.method == 'GAT':
            net = gnn.GAT(
                n_features, args.hidden, args.n_layers, n_labels,
                v2=True, dropout=args.dropout)
        else:
            # from models import GCN
            net = {
                'GIN': gnn.GIN,
                'GCN': gnn.GCN,
                'SAGE': gnn.GraphSAGE,
                'GCNII': GCNII,
            }[args.method](
                n_features, args.hidden, args.n_layers, n_labels, args.dropout)
        net = gpu(net)
        opt = Optim([*net.parameters()])
        if run == 0:
            print('params: 0' if opt is None else opt)

        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                z = net(X, E)
                criterion(z[train_mask], Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)

            # Inference
            t = time.time()
            with torch.no_grad():
                net.eval()
                state = sg(net(X, E))
                net.train()
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(state):
                break
            # print('epoch:', epoch, 'score:', ev.scores[-1])

    ev.end_run()
ev.end_all()
print('script time:', time.time() - script_time)
