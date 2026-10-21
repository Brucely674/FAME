from .Attention import *
from ..utility.CustomGeometry import get_spherical_coords
from ..utility.spherical_harmonics import clear_spherical_harmonics_cache, get_spherical_harmonics

class SconeVis(nn.Module):

    def __init__(self, pts_dim=4, seq_len=2048, pts_embedding_dim=256, n_heads=4, n_code=3, n_harmonics=64, max_harmonic_rank=8, FF=True, gelu=True, dropout=None, use_view_state=True, use_global_feature=True, view_state_mode='end', concatenate_input=True, k_for_knn=0, alt=False, use_sigmoid=True):
        super(SconeVis, self).__init__()
        self.n_harmonics = n_harmonics
        self.pts_dim = pts_dim
        self.seq_len = seq_len
        self.pts_embedding_dim = pts_embedding_dim
        self.n_heads = n_heads
        self.n_code = n_code
        self.n_harmonics = n_harmonics
        self.max_harmonic_rank = max_harmonic_rank
        self.use_view_state = use_view_state
        self.use_global_feature = use_global_feature
        self.view_state_mode = view_state_mode
        self.alt = alt
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            print('Use sigmoid in model.')
        else:
            print('Use ReLU for output in model.')
        if use_view_state and view_state_mode == 'start':
            additional_feature_dim = n_harmonics
        else:
            additional_feature_dim = 0
        self.embedding = Embedding(pts_dim, pts_embedding_dim, gelu=gelu, global_feature=use_global_feature, additional_feature_dim=additional_feature_dim, concatenate_input=concatenate_input, k_for_knn=k_for_knn, dropout=None)
        encoders = []
        for i in range(n_code):
            encoders += [Encoder(seq_len=seq_len, embedding_dim=pts_embedding_dim, qk_dim=pts_embedding_dim // 4, n_heads=n_heads, dropout=dropout, gelu=gelu, FF=FF)]
        self.encoders = nn.ModuleList(encoders)
        self.norm = nn.LayerNorm(pts_embedding_dim)
        if not alt:
            fc1_input_dim = pts_embedding_dim
            inner_feature_factor = 4
            if use_view_state and view_state_mode == 'end':
                inner_feature_factor = 3
        else:
            fc1_input_dim = pts_embedding_dim + n_harmonics
            inner_feature_factor = 4
        self.fc1 = nn.Linear(fc1_input_dim, inner_feature_factor * n_harmonics)
        self.nonlinear1 = nn.GELU()
        self.fc2 = nn.Linear(4 * n_harmonics, 2 * n_harmonics)
        self.nonlinear2 = nn.GELU()
        self.fc3 = nn.Linear(2 * n_harmonics, n_harmonics)

    def forward(self, pts, mask=None, view_harmonics=None):
        n_clouds = len(pts)
        seq_len = pts.shape[1]
        if self.use_view_state and self.view_state_mode == 'start':
            x = self.embedding(pts, additional_feature=view_harmonics)
        else:
            x = self.embedding(pts)
        for encoder in self.encoders:
            x = encoder(x, mask=mask)
        res = self.norm(x)
        if not self.alt:
            res = self.nonlinear1(self.fc1(res))
            if self.use_view_state and self.view_state_mode == 'end':
                res = torch.cat((res, view_harmonics), dim=-1)
            res = self.nonlinear2(self.fc2(res))
            res = self.fc3(res)
        else:
            res = torch.cat((res, view_harmonics), dim=-1)
            res = self.nonlinear1(self.fc1(res))
            res = self.nonlinear2(self.fc2(res))
            res = self.fc3(res)
        res = res.view(n_clouds, seq_len, self.n_harmonics)
        return res

    def compute_visibilities(self, pts, harmonics, X_cam):
        clear_spherical_harmonics_cache()
        n_clouds = pts.shape[0]
        seq_len = pts.shape[1]
        n_harmonics = self.n_harmonics
        n_camera_candidates = X_cam.shape[1]
        device = pts.get_device()
        if device < 0:
            device = 'cpu'
        X_pts = pts[..., :3]
        rays = (X_cam.view(n_clouds, n_camera_candidates, 1, 3).expand(-1, -1, seq_len, -1) - X_pts.view(n_clouds, 1, seq_len, 3).expand(-1, n_camera_candidates, -1, -1)).view(-1, 3)
        _, theta, phi = get_spherical_coords(rays)
        theta = -theta + np.pi / 2.0
        z = torch.zeros([i for i in theta.shape] + [0], device=device)
        for i in range(self.max_harmonic_rank):
            y = get_spherical_harmonics(l=i, theta=theta, phi=phi)
            z = torch.cat((z, y), dim=-1)
        z = z.view(n_clouds, n_camera_candidates, seq_len, n_harmonics)
        z = torch.sum(z * harmonics.view(n_clouds, 1, seq_len, 64).expand(-1, n_camera_candidates, -1, -1), dim=-1)
        if self.use_sigmoid:
            z = torch.sigmoid(z)
        else:
            z = torch.relu(z)
        return z

    def compute_coverage_gain(self, pts, harmonics, X_cam):
        clear_spherical_harmonics_cache()
        n_clouds = pts.shape[0]
        seq_len = pts.shape[1]
        n_harmonics = self.n_harmonics
        n_camera_candidates = X_cam.shape[1]
        X_pts = pts[..., :3]
        rays = (X_cam.view(n_clouds, n_camera_candidates, 1, 3).expand(-1, -1, seq_len, -1) - X_pts.view(n_clouds, 1, seq_len, 3).expand(-1, n_camera_candidates, -1, -1)).view(-1, 3)
        _, theta, phi = get_spherical_coords(rays)
        theta = -theta + np.pi / 2.0
        z = torch.zeros([i for i in theta.shape] + [0], device=theta.get_device())
        for i in range(self.max_harmonic_rank):
            y = get_spherical_harmonics(l=i, theta=theta, phi=phi)
            z = torch.cat((z, y), dim=-1)
        z = z.view(n_clouds, n_camera_candidates, seq_len, n_harmonics)
        z = torch.sum(z * harmonics.view(n_clouds, 1, seq_len, 64).expand(-1, n_camera_candidates, -1, -1), dim=-1)
        if self.use_sigmoid:
            z = torch.sigmoid(z)
        else:
            z = torch.relu(z)
        z = torch.sum(z, dim=-1) / seq_len
        return z

    def compute_coverage_gain_multiple(self, pts, harmonics, X_cam, n_cam):
        clear_spherical_harmonics_cache()
        n_clouds = pts.shape[0]
        seq_len = pts.shape[1]
        n_harmonics = self.n_harmonics
        n_camera_candidates = X_cam.shape[1]
        X_pts = pts[..., :3]
        rays = (X_cam.view(n_clouds, n_camera_candidates, 1, 3).expand(-1, -1, seq_len, -1) - X_pts.view(n_clouds, 1, seq_len, 3).expand(-1, n_camera_candidates, -1, -1)).view(-1, 3)
        _, theta, phi = get_spherical_coords(rays)
        theta = -theta + np.pi / 2.0
        z = torch.zeros([i for i in theta.shape] + [0], device=theta.get_device())
        for i in range(self.max_harmonic_rank):
            y = get_spherical_harmonics(l=i, theta=theta, phi=phi)
            z = torch.cat((z, y), dim=-1)
        z = z.view(n_clouds, n_camera_candidates, seq_len, n_harmonics)
        z = torch.sum(z * harmonics.view(n_clouds, 1, seq_len, 64).expand(-1, n_camera_candidates, -1, -1), dim=-1)
        if self.use_sigmoid:
            z = torch.sigmoid(z)
        else:
            z = torch.relu(z)
        single_idx = torch.arange(0, n_camera_candidates)
        if n_cam == 2:
            n_idx = torch.cartesian_prod(single_idx, single_idx)
        elif n_cam == 3:
            n_idx = torch.cartesian_prod(single_idx, single_idx, single_idx)
        else:
            raise NameError('n_cam is too large.')
        n_z = z[:, n_idx]
        n_z = torch.sum(torch.max(n_z, dim=-2)[0], dim=-1) / seq_len
        return (n_z, n_idx)

class KLDivCE(nn.Module):

    def __init__(self):
        super(KLDivCE, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.log_soft_max = nn.LogSoftmax(dim=1)

    def forward(self, x, y):
        loss = self.kl_div(self.log_soft_max(x), torch.softmax(y, dim=1))
        return loss

class L1_loss(nn.Module):

    def __init__(self):
        super(L1_loss, self).__init__()
        self.epsilon = 1e-07

    def forward(self, x, y):
        batch_size, n_camera = (x.shape[0], x.shape[1])
        x_mean = x.mean(dim=1, keepdim=True).expand(-1, n_camera, -1)
        y_mean = y.mean(dim=1, keepdim=True).expand(-1, n_camera, -1)
        x_std = x.std(dim=1, keepdim=True).expand(-1, n_camera, -1)
        y_std = y.std(dim=1, keepdim=True).expand(-1, n_camera, -1)
        norm_x = (x - x_mean) / (x_std + self.epsilon)
        norm_y = (y - y_mean) / (y_std + self.epsilon)
        loss = (norm_x - norm_y).abs().mean(dim=1)
        return loss.mean()

class Uncentered_L1_loss(nn.Module):

    def __init__(self):
        super(Uncentered_L1_loss, self).__init__()
        self.epsilon = 1e-07

    def forward(self, x, y):
        batch_size, n_camera = (x.shape[0], x.shape[1])
        x_mean = x.mean(dim=1, keepdim=True).expand(-1, n_camera, -1)
        y_mean = y.mean(dim=1, keepdim=True).expand(-1, n_camera, -1)
        norm_x = x / (x_mean + self.epsilon)
        norm_y = y / (y_mean + self.epsilon)
        loss = (norm_x - norm_y).abs().mean(dim=1)
        return loss.mean()
