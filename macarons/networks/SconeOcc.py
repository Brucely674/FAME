import torch
import torch.nn.functional as F
from .Attention import *
from ..utility.utils import get_knn_points

class XEmbedding(nn.Module):

    def __init__(self, x_dim, x_embedding_dim, dropout=None, gelu=True):
        super(XEmbedding, self).__init__()
        self.linear1 = nn.Linear(x_dim, x_embedding_dim // 4)
        self.linear2 = nn.Linear(x_embedding_dim // 4, x_embedding_dim // 2)
        self.linear3 = nn.Linear(x_embedding_dim // 2, x_embedding_dim)
        if gelu:
            self.non_linear1 = nn.GELU()
            self.non_linear2 = nn.GELU()
            self.non_linear3 = nn.GELU()
        else:
            self.non_linear1 = nn.ReLU(inplace=False)
            self.non_linear2 = nn.ReLU(inplace=False)
            self.non_linear3 = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        res = self.non_linear1(self.linear1(x))
        res = self.non_linear2(self.linear2(res))
        res = self.non_linear3(self.linear3(res))
        res = res if self.dropout is None else self.dropout(res)
        return res

class PCTransformer(nn.Module):

    def __init__(self, seq_len, pts_dim=3, pts_embedding_dim=256, feature_dim=512, concatenate_input=True, n_code=2, n_heads=4, FF=True, gelu=True, dropout=None):
        super(PCTransformer, self).__init__()
        self.seq_len = seq_len
        self.pts_dim = pts_dim
        self.pts_embedding_dim = pts_embedding_dim
        self.n_code = n_code
        self.n_heads = n_heads
        self.FF = FF
        self.gelu = gelu
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.embedding = Embedding(input_dim=pts_dim, output_dim=pts_embedding_dim, dropout=None, gelu=gelu, global_feature=False, additional_feature_dim=0, concatenate_input=concatenate_input, k_for_knn=0)
        encoders = []
        for i in range(n_code):
            encoders += [Encoder(seq_len=seq_len, embedding_dim=pts_embedding_dim, qk_dim=pts_embedding_dim // 4, n_heads=n_heads, dropout=dropout, gelu=gelu, FF=FF)]
        self.encoders = nn.ModuleList(encoders)
        self.norm = nn.LayerNorm(self.pts_embedding_dim)
        self.linear0 = nn.Linear(self.pts_embedding_dim, feature_dim // 2)
        self.max_pool = nn.functional.max_pool1d
        self.avg_pool = nn.functional.avg_pool1d

    def forward(self, pc, mask=None):
        n_clouds, seq_len = (pc.shape[0], pc.shape[1])
        pc_embd = self.embedding(pc)
        for encoder in self.encoders:
            pc_embd = encoder(pc_embd, mask=mask)
        features = self.norm(pc_embd)
        features = self.linear0(features)
        features = features.transpose(dim0=-1, dim1=-2)
        features = torch.cat((self.max_pool(input=features, kernel_size=seq_len), self.avg_pool(input=features, kernel_size=seq_len)), dim=-2)
        features = features.view(n_clouds, self.feature_dim)
        return features

class SconeOcc(nn.Module):

    def __init__(self, seq_len=2048, pts_dim=3, pts_embedding_dim=128, concatenate_input=True, n_code=2, n_heads=4, FF=True, gelu=True, global_feature_dim=512, n_scale=3, local_feature_dim=256, k_for_knn=16, x_dim=3, x_embedding_dim=512, n_harmonics=64, output_dim=1, dropout=None, offset=True):
        super(SconeOcc, self).__init__()
        self.seq_len = seq_len
        self.pts_dim = pts_dim
        self.pts_embedding_dim = pts_embedding_dim
        self.n_code = n_code
        self.n_heads = n_heads
        self.FF = FF
        self.gelu = gelu
        self.n_scale = n_scale
        self.x_dim = x_dim
        self.x_embedding_dim = x_embedding_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.encoding_dim = pts_embedding_dim
        self.k_for_knn = k_for_knn
        self.offset = offset
        if self.offset:
            print('Offset set to True.')
        self.global_feature_dim = global_feature_dim
        self.local_feature_dim = local_feature_dim
        self.all_feature_size = self.x_embedding_dim + self.n_scale * self.local_feature_dim + self.global_feature_dim + n_harmonics
        self.global_transformer = PCTransformer(seq_len=seq_len, pts_dim=pts_dim, pts_embedding_dim=pts_embedding_dim, feature_dim=global_feature_dim, concatenate_input=concatenate_input, n_code=n_code, n_heads=n_heads, FF=FF, gelu=gelu, dropout=dropout)
        local_transformers = []
        for i in range(n_scale):
            local_transformers += [PCTransformer(seq_len=k_for_knn, pts_dim=pts_dim, pts_embedding_dim=pts_embedding_dim, feature_dim=local_feature_dim, concatenate_input=concatenate_input, n_code=n_code, n_heads=n_heads, FF=FF, gelu=gelu, dropout=dropout)]
        self.local_transformers = nn.ModuleList(local_transformers)
        self.x_embedding = XEmbedding(x_dim=x_dim, x_embedding_dim=x_embedding_dim, dropout=dropout, gelu=gelu)
        self.max_pool = nn.functional.max_pool1d
        self.avg_pool = nn.functional.avg_pool1d
        self.linear1 = nn.Linear(self.all_feature_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, output_dim)
        if gelu:
            self.non_linear1 = nn.GELU()
            self.non_linear2 = nn.GELU()
            self.non_linear3 = nn.GELU()
        else:
            self.non_linear1 = nn.ReLU(inplace=False)
            self.non_linear2 = nn.ReLU(inplace=False)
            self.non_linear3 = nn.ReLU(inplace=False)

    def forward(self, pc, x, view_harmonics, mask=None, verbose=False):
        n_clouds, full_seq_len = (pc.shape[0], pc.shape[1])
        n_sample = x.shape[1]
        if verbose:
            with torch.no_grad():
                print('\nInput:', torch.norm(pc))
        global_down_sampled_pc = pc[:, torch.randperm(pc.shape[1])[:self.seq_len]]
        seq_len = global_down_sampled_pc.shape[1]
        if verbose:
            with torch.no_grad():
                print('Global ds pc:', torch.norm(global_down_sampled_pc))
        global_features = self.global_transformer(global_down_sampled_pc)
        if verbose:
            with torch.no_grad():
                print('Global features:', torch.norm(global_features))
        if self.n_scale > 1:
            ds_factor = int(np.power(full_seq_len / (self.k_for_knn * 8), 1.0 / (self.n_scale - 1)))
            if ds_factor == 0:
                ds_factor = 2
        else:
            ds_factor = 1
        down_sampled_pc = pc
        local_transformed = []
        for n_transformer in range(self.n_scale):
            local_transformer = self.local_transformers[n_transformer]
            local_pc, _, _ = get_knn_points(x, down_sampled_pc, self.k_for_knn)
            if self.offset:
                local_pc = local_pc - x.view(n_clouds, n_sample, 1, 3)
            local_transformed += [local_transformer(local_pc.view(-1, self.k_for_knn, 3), mask=mask)]
            if verbose:
                with torch.no_grad():
                    print('Local pc:', torch.norm(local_pc), ';  Local transformed:', torch.norm(local_transformed[-1]))
            ds_seq_len = down_sampled_pc.shape[1]
            if n_transformer < self.n_scale - 1:
                down_sampled_pc = down_sampled_pc[:, torch.randperm(ds_seq_len)[:ds_seq_len // ds_factor]]
        if self.n_scale > 0:
            local_features = torch.cat(local_transformed, dim=-1)
        else:
            local_features = torch.zeros(n_clouds, n_sample, 0, device=pc.get_device())
        local_features = local_features.view(n_clouds, n_sample, self.n_scale * self.local_feature_dim)
        if verbose:
            with torch.no_grad():
                print('Local features:', torch.norm(local_features))
        x_features = self.x_embedding(x)
        if verbose:
            with torch.no_grad():
                print('X encoding:', torch.norm(x_features))
        global_features = global_features.view(n_clouds, 1, self.global_feature_dim).expand(-1, n_sample, -1)
        x_features = x_features.view(n_clouds, n_sample, self.x_embedding_dim)
        res = torch.cat((global_features, local_features, x_features, view_harmonics), dim=-1)
        res = self.non_linear1(self.linear1(res))
        if verbose:
            with torch.no_grad():
                print('Linear 1:', torch.norm(res))
        res = self.non_linear2(self.linear2(res))
        if verbose:
            with torch.no_grad():
                print('Linear 2:', torch.norm(res))
        res = self.non_linear3(self.linear3(res))
        if verbose:
            with torch.no_grad():
                print('Linear 3:', torch.norm(res))
        return res.view(n_clouds, n_sample, self.output_dim)
