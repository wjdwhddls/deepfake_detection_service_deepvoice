import torch
import torch.nn as nn

# Network that transforms fixed-length embedding
class EmbeddingTransformNetwork(nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim):
        super(EmbeddingTransformNetwork, self).__init__()

        self.fc_input = nn.Linear(emb_dim, hidden_dim)
        self.fc_hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden_4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)
        self.bn_3 = nn.BatchNorm1d(hidden_dim)
        self.bn_4 = nn.BatchNorm1d(hidden_dim)
        self.bn_5 = nn.BatchNorm1d(hidden_dim)

        self.leaky_relu = nn.LeakyReLU()

        return

    def forward(self, x):
        x = self.fc_input(x)
        x = self.leaky_relu(x)
        x = self.bn_1(x)

        x = self.fc_hidden_1(x)
        x = self.leaky_relu(x)
        x = self.bn_2(x)

        x = self.fc_hidden_2(x)
        x = self.leaky_relu(x)
        x = self.bn_3(x)

        x = self.fc_hidden_3(x)
        x = self.leaky_relu(x)
        x = self.bn_4(x)

        x = self.fc_hidden_4(x)
        x = self.leaky_relu(x)
        x = self.bn_5(x)

        x = self.fc_out(x)
        x = self.leaky_relu(x)

        return x

# Siamese network that transforms each embedding and calculate cosine similarity between two embedding
class SiamCosSimNetwork(nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim):
        super(SiamCosSimNetwork, self).__init__()

        self.emb_ts_1 = EmbeddingTransformNetwork(emb_dim, hidden_dim, hidden_dim)
        self.emb_ts_2 = EmbeddingTransformNetwork(emb_dim, hidden_dim, hidden_dim)

        self.fc_hidden_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.bn_emb_1 = nn.BatchNorm1d(hidden_dim)
        self.bn_emb_2 = nn.BatchNorm1d(hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, anchor, ref):
        # Apply transformation to anchor embedding
        x_anchor = self.emb_ts_1(anchor)
        x_anchor = self.leaky_relu(x_anchor)

        # Apply transformation to reference embedding
        x_ref = self.emb_ts_2(ref)
        x_ref = self.leaky_relu(x_ref)

        # Calculate Cosine Similarity
        cosine_score = torch.mul(x_anchor, x_ref).sum(dim = 1) / torch.mul(torch.norm(x_anchor, dim = 1), torch.norm(x_ref, dim = 1))

        # Rescale the similarity, from [-1, 1] to [0, 1]
        cosine_score = (cosine_score.unsqueeze(-1) + 1.0) / 2

        return cosine_score

# Just the mixture of two siamese network
class SiamMetricNetworks(nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim):
        super(SiamMetricNetworks, self).__init__()

        self.fake_dist_net = SiamCosSimNetwork(emb_dim, hidden_dim, out_dim)
        self.real_dist_net = SiamCosSimNetwork(emb_dim, hidden_dim, out_dim)

        return

    def forward(self, anchor, ref_fake, ref_real):
        fake_dist = self.fake_dist_net(anchor, ref_fake)
        real_dist = self.real_dist_net(anchor, ref_real)

        return fake_dist, real_dist

# Testing with dummy input
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SiamMetricNetworks(256, 256, 1).to(device)
    dummy_anchor, dummy_fake, dummy_real = torch.randn(3, 64, 256).to(device)

    fake_dist, real_dist = model(dummy_anchor, dummy_fake, dummy_real)

    print(f'Fake Dist: {fake_dist.shape}\nReal Dist: {real_dist.shape}')