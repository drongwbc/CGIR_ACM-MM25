import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering


class GeneratorModel(nn.Module):
    def __init__(self, num_features, hidden_size, emblem_size, sub_clusters):
        super(GeneratorModel, self).__init__()
        # GCN
        self.gcn_l1 = nn.Linear(num_features, hidden_size, bias=False)
        self.gcn_l2 = nn.Linear(hidden_size, emblem_size, bias=False)

        self.mvn = torch.distributions.MultivariateNormal(
            torch.zeros(emblem_size), torch.eye(emblem_size))

        # Edge Attention Network
        self.ean_l1 = EAN(emblem_size, int(emblem_size / 2))
        self.ean_l2 = EAN(int(emblem_size / 2), int(emblem_size / 4))

    def calculate_P(self, Z, centroid, v=1):
        device = Z.device
        p = 1.0 / (1.0 + torch.sum(
            torch.pow(Z.unsqueeze(1).to(device) - centroid, 2), 2) / v)
        p = p.pow((v + 1.0) / 2.0)
        p = (p.t() / torch.sum(p, 1).to(device)).t()
        p = torch.clamp(p, min=1e-12)
        return p

    def subcluster_aware(self, Z, sub_clusters):
        agg_clustering = AgglomerativeClustering(
            n_clusters=sub_clusters, linkage='average')
        new_labels = agg_clustering.fit_predict(Z.cpu().detach().numpy())
        centroid = torch.zeros((sub_clusters, Z.size(1)), device=Z.device)
        for i in range(sub_clusters):
            cluster_points = Z[new_labels == i]
            if cluster_points.size(0) > 0:
                centroid[i] = cluster_points.mean(dim=0)

        p = self.calculate_P(Z, centroid)
        return p, centroid

    def subcluster_agglomerative(self, Z, P, n_clusters):
        centroids = torch.mm(P.T, Z) / P.sum(dim=0).unsqueeze(1)
        centroids_cpu = centroids.cpu().detach().numpy()
        agg_clustering = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='average')
        new_labels = agg_clustering.fit_predict(centroids_cpu)
        new_P = torch.zeros((P.size(0), n_clusters), device=Z.device)
        for j in range(P.size(1)):
            new_P[:, new_labels[j]] += P[:, j]
        centroid = torch.mm(new_P.T, Z) / new_P.sum(dim=0).unsqueeze(1)
        p = self.calculate_P(Z, centroid)
        return p

    def forward(self, X, S, sub_clusters, n_clusters, drop_indices):
        # Encoder
        Z = F.relu(self.gcn_l1(torch.spmm(S, X)))
        Z = F.relu(self.gcn_l2(torch.spmm(S, Z)))
        Z = F.normalize(Z, p=2, dim=1)
        sub_P, centroids = self.subcluster_aware(Z, sub_clusters)

        # Subcluster Distribution
        column_sums = sub_P.sum(dim=0).unsqueeze(1)
        mu = torch.mm(sub_P.T, Z) / column_sums
        zita = torch.zeros(sub_clusters, Z.size(1), Z.size(1)).to(Z.device)
        for i in range(sub_clusters):
            deviations = Z - centroids[i]
            weighted_cov = (sub_P[:, i].unsqueeze(
                1) * deviations).t() @ deviations
            zita[i] = weighted_cov / column_sums[i]

        # Imputation
        noise = self.mvn.sample((drop_indices.size(0),)).to(Z.device)
        H = torch.zeros(noise.size()).to(Z.device)
        for i in range(sub_clusters):
            H += torch.mm(torch.diag(sub_P[drop_indices][:, i]
                                     ).to(Z.device), mu[i] + torch.mm(noise, zita[i]))

        diag_matrix = torch.eye(Z.size(0), Z.size(0)).to(Z.device)
        diag_matrix[drop_indices, drop_indices] = 0
        H = F.normalize(H, p=2, dim=1)
        Z = (torch.mm(diag_matrix, Z).scatter_(
            dim=0, index=drop_indices.unsqueeze(1).expand(-1, Z.size(1)), src=H) + Z) / 2
        Z = F.normalize(Z, p=2, dim=1)

        # Subcluster Mergence
        P = self.subcluster_agglomerative(Z, sub_P, n_clusters)

        # EAN
        V, A1 = self.ean_l1(Z, S)
        V, A2 = self.ean_l2(V, S)
        A = [A1, A2]
        V = F.normalize(V, p=2, dim=1)

        # Decoder
        rec_S = F.sigmoid(torch.mm(V, V.T))

        return Z, rec_S, mu, zita, P, A, sub_P


class DiscriminatorModel(nn.Module):
    def __init__(self, hidden_size, emblem_size, sub_clusters):
        super(DiscriminatorModel, self).__init__()
        # Discriminator
        self.dis_l1 = nn.Linear(hidden_size, emblem_size, bias=True)
        self.dis_l2 = nn.Linear(emblem_size, sub_clusters + 1, bias=True)

    def forward(self, Z):
        Y = F.relu(self.dis_l1(Z))
        Y = F.softmax(self.dis_l2(Y))

        return Y


class EAN(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(EAN, self).__init__()
        # Edge Attention Network
        self.mlp_q = nn.Linear(num_features, hidden_size, bias=False)
        self.mlp_k = nn.Linear(num_features, hidden_size, bias=False)
        self.mlp_v = nn.Linear(num_features, hidden_size, bias=False)

    def get_A(self, Q, K, S):
        Q = F.normalize(Q, p=2, dim=1)
        K = F.normalize(K, p=2, dim=1)
        S = S.coalesce()
        indices = S.indices()
        row_idx = indices[0]
        col_idx = indices[1]
        Q_selected = Q[row_idx]
        K_selected = K[col_idx]
        hadamard_product = Q_selected * K_selected
        result = torch.zeros_like(Q)
        result.index_add_(0, row_idx, hadamard_product)
        result = result / torch.sparse.sum(S, dim=1).to_dense().unsqueeze(1)
        result = result / (Q.size(1) ** (0.5))
        result = F.softmax(result)
        return result

    def forward(self, Z, S):
        Q = self.mlp_q(Z)
        K = self.mlp_k(Z)
        A = self.get_A(Q, K, S)
        V = torch.mul(self.mlp_v(Z), A)
        return V, A
