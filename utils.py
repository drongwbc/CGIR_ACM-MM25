import torch
import torch.nn.functional as F


def load_incomplete(name):
    path = 'data/{}.pth'.format(name)
    loaded_tensors = torch.load(path, weights_only=False)
    X = loaded_tensors['X']  # Attribute Matrix
    S = loaded_tensors['S']  # Graph
    Y = loaded_tensors['Y']  # True Label
    # Index for Attribute-missing Node
    drop_indices = loaded_tensors['drop_indices']

    return X, S, Y, drop_indices


def graph_normalize(S, isLoop=True):
    """
        Normalize the adj matrix
    """
    if isLoop:
        S.fill_diagonal_(1)

    degree = torch.sum(S, dim=1).pow(-0.5)
    return (S * degree).t() * degree


def gen_discrimination_loss(Y, sub_P):
    labels = sub_P.argmax(dim=1)
    loss = F.cross_entropy(Y, labels)
    return loss


def dis_discrimination_loss(Y, sub_P, drop_indices):
    num_classes = Y.size(1)
    labels = sub_P.argmax(dim=1)
    for index in drop_indices:
        labels[index] = num_classes - 1

    # 计算交叉熵损失
    loss = F.cross_entropy(Y, labels)
    return loss


def earth_movers_distance(mu1, sigma1, mu2, sigma2):
    mean_distance = torch.norm(mu1 - mu2)

    sigma1_sqrt = torch.linalg.cholesky(sigma1)  # 计算协方差矩阵的 Cholesky 分解
    sigma1_inv_sqrt = torch.linalg.inv(sigma1_sqrt)  # 计算逆
    sigma1_inv_sqrt_sigma2 = sigma1_inv_sqrt @ sigma2 @ sigma1_inv_sqrt  # 矩阵乘法

    sigma1_inv_sqrt_sigma2_sqrt = torch.linalg.cholesky(sigma1_inv_sqrt_sigma2)

    cov_distance = torch.trace(
        sigma1 + sigma2 - 2 * sigma1_inv_sqrt_sigma2_sqrt)

    emd = mean_distance + cov_distance
    return emd


def subcluster_aware_loss(mu, zita):
    loss = 0
    c = mu.size(0)
    for i in range(c):
        zita[i] = zita[i] + torch.eye(mu.size(1)).to(mu.device)

    for i in range(c):
        loss += torch.det(zita[i])

    return loss / c


def contrastive_loss(A, P, tau=0.1):
    labels = torch.argmax(P, dim=1)

    A = F.normalize(A, p=2, dim=1)
    sim_matrix = torch.mm(A, A.T) / tau

    positive_mask = (labels.unsqueeze(
        1) == labels.unsqueeze(0)).float()

    positive_scores = torch.exp(
        torch.mul(sim_matrix, positive_mask))
    negative_scores = torch.exp(
        torch.mul(sim_matrix, 1 - positive_mask))

    pos_sum = positive_scores.sum()
    neg_sum = negative_scores.sum()

    loss = -torch.log(pos_sum / (pos_sum + neg_sum)) / (2 * (P.size(0)))

    return loss


def graph_reconstruction_loss(S, rec_S):
    n = S.size(0)
    loss = - (S * rec_S.log() + (1 - S) * (1 - rec_S).log()).sum() / (n * n)
    return loss
