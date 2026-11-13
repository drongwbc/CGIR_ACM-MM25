import torch
import utils
import numpy as np
import random
import warnings
from model import GeneratorModel, DiscriminatorModel
from torch.optim import Adam
from sklearn.cluster import KMeans
from evaluation import eva


def setup_seed(seed):
    if seed == 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    setup_seed(args.seed)
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    X, S, gt, drop_indices = utils.load_incomplete(args.name)
    X = X.to(device)
    S = S.to(device)
    drop_indices = drop_indices.to(device)

    args.samples = X.shape[0]
    args.input_dim = X.shape[1]
    args.n_clusters = len(np.unique(gt))
    args.sub_clusters = args.n_clusters * 4
    nor_S = utils.graph_normalize(S).to_sparse()

    print('---------Dataset Details---------')
    print('name: ' + args.name)
    print(f'samples: {args.samples:.0f}')
    print(f'cluster: {args.n_clusters:.0f}')
    print(f'dimension: {X.shape[1]:.0f} ', end='')
    print('\n---------------------------------')

    GenModel = GeneratorModel(num_features=args.input_dim,
                              hidden_size=args.hidden_size,
                              emblem_size=args.emblem_size,
                              sub_clusters=args.sub_clusters).to(device)

    DisModel = DiscriminatorModel(hidden_size=args.emblem_size,
                                  emblem_size=int(args.emblem_size / 4),
                                  sub_clusters=args.sub_clusters).to(device)

    optimizer_gen = Adam(GenModel.parameters(),
                         lr=args.lr,
                         weight_decay=args.weight_decay)

    optimizer_dis = Adam(DisModel.parameters(),
                         lr=args.lr,
                         weight_decay=args.weight_decay)

    retScore = [0, 0, 0, 0]
    for epoch in range(args.max_epoch):
        GenModel.train()
        DisModel.train()
        Z, rec_S, mu, zita, P, A, sub_P = GenModel(
            X, nor_S, args.sub_clusters, args.n_clusters, drop_indices)
        Y = DisModel(Z)

        # Train Discriminator
        d_loss = utils.dis_discrimination_loss(Y, sub_P, drop_indices)

        optimizer_dis.zero_grad()
        d_loss.backward(retain_graph=True)

        # Train Generator
        l1 = utils.subcluster_aware_loss(mu, zita)
        l2 = utils.gen_discrimination_loss(Y, sub_P)
        l3 = (utils.contrastive_loss(A[0], P) +
              utils.contrastive_loss(A[1], P)) / 2
        l4 = utils.graph_reconstruction_loss(S, rec_S)
        g_loss = l1 + args.lambda1*l2 + args.lambda2*l3 + args.lambda3*l4

        optimizer_gen.zero_grad()
        g_loss.backward()
        optimizer_dis.step()
        optimizer_gen.step()

        with torch.no_grad():
            GenModel.eval()
            DisModel.eval()
            label = KMeans(n_clusters=args.n_clusters, n_init=1,
                           init='k-means++').fit(Z.detach().cpu().numpy()).labels_

            acc, nmi, ari, f1 = eva(gt, label)
            retScore[0] = max(retScore[0], acc)
            retScore[1] = max(retScore[1], nmi)
            retScore[2] = max(retScore[2], ari)
            retScore[3] = max(retScore[3], f1)
            print(
                f'epoch {epoch}: acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}, loss: {g_loss.item():.4f}')

    print("---------------------")
    print(
        f'final acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}')
    print(
        f'best  acc: {retScore[0]:.4f}, nmi: {retScore[1]:.4f}, ari: {retScore[2]:.4f}, f1: {retScore[3]:.4f}'
    )

    return
