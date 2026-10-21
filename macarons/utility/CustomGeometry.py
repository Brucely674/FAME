import numpy as np
import torch

def get_cartesian_coords(r, elev, azim, in_degrees=False):
    factor = 1
    if in_degrees:
        factor *= np.pi / 180.0
    X = torch.stack((torch.cos(factor * elev) * torch.sin(factor * azim), torch.sin(factor * elev), torch.cos(factor * elev) * torch.cos(factor * azim)), dim=2)
    return r * X.view(-1, 3)

def get_spherical_coords(X):
    r_x = torch.linalg.norm(X, dim=1)
    elev_x = torch.asin(X[:, 1] / r_x)
    elev_x[X[:, 1] / r_x <= -1] = -np.pi / 2
    elev_x[X[:, 1] / r_x >= 1] = np.pi / 2
    azim_x = torch.acos(X[:, 2] / (r_x * torch.cos(elev_x)))
    azim_x[X[:, 2] / (r_x * torch.cos(elev_x)) <= -1] = np.pi
    azim_x[X[:, 2] / (r_x * torch.cos(elev_x)) >= 1] = 0.0
    azim_x[X[:, 0] < 0] *= -1
    return (r_x, elev_x, azim_x)

def sample_cameras_on_sphere(n_X, radius, device):
    delta_theta = 0.9 * np.pi
    delta_phi = 0.9 * 2 * np.pi
    n_dim = int(np.sqrt(n_X))
    d_theta = 2 * delta_theta / (n_dim - 1)
    d_phi = 2 * delta_phi / (n_dim - 1)
    increments = torch.linspace(0, n_dim - 1, n_dim, device=device)
    thetas = -delta_theta + increments * d_theta
    phis = -delta_phi + increments * d_phi
    thetas = thetas.view(n_dim, 1).expand(-1, n_dim)
    phis = phis.view(1, n_dim).expand(n_dim, -1)
    X = torch.stack((torch.cos(thetas) * torch.sin(phis), torch.sin(thetas), torch.cos(thetas) * torch.cos(phis)), dim=2)
    return radius * X.view(-1, 3)

def dot_prod(a, b, keepdim=False):
    return (a * b).sum(-1, keepdim=keepdim)
