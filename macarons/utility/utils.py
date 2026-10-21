import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import os
from collections import OrderedDict
import re
import json
import time
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.ops import knn_gather
from pytorch3d.structures import Meshes
from pytorch3d.datasets import collate_batched_meshes, render_cubified_voxels
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, FoVOrthographicCameras, Materials, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex, TexturesAtlas, PointsRenderer, PointsRasterizationSettings, PointsRasterizer, OpenGLPerspectiveCameras, PointLights, Textures, softmax_rgb_blend, BlendParams
from pytorch3d.renderer.mesh.shading import flat_shading
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from .CustomGeometry import get_cartesian_coords

def flatten_dict(d, d_out):
    for key, v in d.items():
        if key[0] == '_':
            flatten_dict(d[key], d_out)
        else:
            d_out[key] = d[key]

class Params:

    def __init__(self, json_path, flatten=False):
        with open(json_path) as f:
            params = json.load(f)
            if flatten:
                d_out = {}
                flatten_dict(params, d_out)
                params = d_out
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

class TimeCheck:

    def __init__(self, unit='seconds'):
        flags = []
        flag_names = []
        self.unit = unit

    def start(self):
        self.flags = [time.time()]
        self.flag_names = ['start']

    def flag(self, flag_name=None):
        self.flags.append(time.time())
        if flag_name is None:
            flag_name = 'no name'
        self.flag_names.append(flag_name)

    def print_flags(self):
        print('\n-----Time flags-----')
        for i in range(1, len(self.flags)):
            value = self.flags[i] - self.flags[i - 1]
            if self.unit == 'minutes':
                value /= 60.0
            if self.unit == 'hours':
                value /= 3600.0
            print(self.flag_names[i] + ':', value)
        print('----------\n')

def floor_divide(x, d):
    res = (x - x % d) / d
    return res

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

def init_weights_selu(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

def load_ddp_state_dict(model, state_dict):
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in state_dict.items():
        if re.search('module', k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict)
    return model

def load_weights(model, trained_weights_file, ddp_model, device):
    model = model.to(device)
    checkpoint = torch.load(trained_weights_file, map_location=device)
    print('Model name:', trained_weights_file)
    print('Trained for', checkpoint['epoch'], 'epochs.')
    print('Training finished with loss', checkpoint['loss'])
    if ddp_model:
        model = load_ddp_state_dict(model, checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def check_gradients(model):
    total_norm = 0.0
    print('\n-----Gradients norm-----')
    for name, p in model.named_parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2).item()
            print(name + ':', param_norm)
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    print('Total norm:', total_norm, '\n')

class FastTensorDataLoader:

    def __init__(self, batch_size, auto_shuffle=False, dataset=None, **tensors):
        assert all((tensors[key].shape[0] == tensors[list(tensors.keys())[0]].shape[0] for key in tensors))
        self.tensors = tensors
        self.dataset_len = self.tensors[list(self.tensors.keys())[0]].shape[0]
        self.batch_size = batch_size
        self.auto_shuffle = auto_shuffle
        if dataset is not None:
            self.dataset = dataset
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.auto_shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i + self.batch_size]
            batch = {key: torch.index_select(self.tensors[key], 0, indices) for key in self.tensors}
        else:
            batch = {key: self.tensors[key][self.i:self.i + self.batch_size] for key in self.tensors}
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    def shuffle(self):
        indices = torch.randperm(self.dataset_len)
        self.tensors = {key: self.tensors[key][indices] for key in self.tensors}

class NoamOpt:

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))

class WarmupConstantOpt:

    def __init__(self, learning_rate, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.learning_rate = learning_rate
        self._rate = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.learning_rate * min(1.0, step / self.warmup)

class WarmupExponentialOpt:

    def __init__(self, start_lr, end_lr, warmup, decay, optimizer, start_factor=0.01, begin_after=0):
        self.optimizer = optimizer
        self._step = 0
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.warmup = warmup
        self.decay = decay
        self.start_factor = start_factor
        self.begin_after = begin_after
        self._rate = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step - self.begin_after
        if step >= 0:
            if self.warmup > 0:
                factor = self.start_factor + (1 - self.start_factor) * np.sin(np.pi / 2 * np.clip(step / self.warmup, 0, 1))
            else:
                factor = 1.0
            return factor * np.exp((1 - step / self.decay) * np.log(self.start_lr) + step / self.decay * np.log(self.end_lr))
        else:
            return 0.0

def image_grid(images, rows=None, cols=None, fill: bool=True, show_axes: bool=False, rgb: bool=True, figsize=(15, 9), titles=None):
    if (rows is None) != (cols is None):
        raise ValueError('Specify either both rows and cols or neither.')
    if rows is None:
        rows = len(images)
        cols = 1
    gridspec_kw = {'wspace': 0.0, 'hspace': 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=figsize)
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=1 - bleed, top=1 - bleed)
    i = 0
    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            ax.imshow(im[..., :3])
        else:
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
        if titles is not None:
            ax.set_title(titles[i])
            i += 1

def custom_zip(coords_per_pixel, image_size=512, z_max=10.0):
    pts = []
    img_coords = []
    device = coords_per_pixel.get_device()
    xpix = torch.Tensor([[i for j in range(image_size)] for i in range(image_size)]).to(device).long()
    ypix = torch.Tensor([[j for j in range(image_size)] for i in range(image_size)]).to(device).long()
    pix = image_size * xpix + ypix
    for i in range(len(coords_per_pixel)):
        mask = coords_per_pixel[i, ..., 2] < z_max
        pts.append(coords_per_pixel[i][mask])
        img_coords.append(pix[mask])
    zipped_coords_per_pixel = {'pts': pts, 'img_coords': img_coords}
    return zipped_coords_per_pixel

def custom_unzip(zipped_coords_per_pixel, image_size=512, z_max=10.0):
    pts = zipped_coords_per_pixel['pts']
    img_coords = zipped_coords_per_pixel['img_coords']
    device = pts[0].get_device()
    coords_per_pixel = z_max * torch.ones(len(pts), image_size, image_size, 3, device=device)
    for i in range(len(pts)):
        coords_per_pixel[i].view(-1, 3)[img_coords[i]] = pts[i]
    return coords_per_pixel

def get_memory_size(zipped_coords_per_pixel):
    size = 0
    for key in zipped_coords_per_pixel.keys():
        for tensor in zipped_coords_per_pixel[key]:
            size += tensor.element_size() * tensor.nelement()
    return size

def remove_heavy_files(file_paths, memory_threshold):
    sizes = []
    for file_path in file_paths:
        sizes.append(os.path.getsize(file_path))
    return np.array(file_paths)[np.array(sizes) < memory_threshold]

def count_parameters(model):
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def create_renderer(params_json_file, device):
    params = Params(params_json_file)
    R, T = look_at_view_transform(dist=params.camera_dist, elev=params.elevation, azim=params.azim_angle)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(image_size=params.image_size, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(device=device, cameras=cameras)
    return MeshRenderer(rasterizer, shader)

def adjust_mesh_ball(vertices):
    min_v = torch.min(vertices)
    max_v = torch.max(vertices)
    mean = torch.mean(vertices, axis=0)
    new_vertices = vertices - mean
    new_vertices /= 2 * torch.max(torch.linalg.norm(new_vertices, axis=1))
    return new_vertices

def compute_bounding_box(vertices):
    min_point = torch.min(vertices, dim=0)[0]
    max_point = torch.max(vertices, dim=0)[0]
    size = max_point - min_point
    return (min_point, max_point, size)

def make_rectangle_meshes(mesh_width, mesh_lengths, device):
    mesh_dict = {}
    mesh_size = mesh_width * 0.5
    base_rectangle = torch.Tensor([[mesh_size, mesh_size, -0.2], [-mesh_size, mesh_size, -0.2], [mesh_size, -mesh_size, -0.2], [-mesh_size, -mesh_size, -0.2], [mesh_size, mesh_size, -0.2], [-mesh_size, mesh_size, -0.2], [mesh_size, -mesh_size, -0.2], [-mesh_size, -mesh_size, -0.2]]).to(device)
    mesh_dict['verts'] = base_rectangle.view(1, 8, 3).expand(len(mesh_lengths), -1, -1) + 0.0
    mesh_dict['verts'][:, :4, 2] += mesh_lengths.view(-1, 1).expand(-1, 4)
    mesh_dict['faces'] = torch.Tensor([[0, 2, 1], [1, 2, 3], [4, 6, 5], [5, 6, 7], [5, 7, 1], [7, 3, 1], [0, 2, 4], [2, 6, 4], [0, 4, 1], [4, 5, 1], [2, 6, 7], [2, 7, 3]]).to(device).view(1, 12, 3).expand(len(mesh_lengths), -1, -1)
    mesh_dict['atlas'] = torch.ones((len(mesh_lengths), 12, 4, 4, 3), device=device)
    mesh_dict['path'] = mesh_lengths.cpu().numpy().astype('str')
    return mesh_dict

def adjust_mesh(vertices, verts_range=1.0):
    device = vertices.get_device()
    if device < 0:
        new_vertices = torch.zeros(vertices.shape)
    else:
        new_vertices = torch.zeros(vertices.shape, device=device)
    max_space_extend = 0.0
    for i in range(3):
        max_coord = torch.max(vertices[:, i])
        min_coord = torch.min(vertices[:, i])
        space_extend = max_coord - min_coord
        new_vertices[:, i] = -space_extend / 2 + vertices[:, i] - min_coord
        if space_extend > max_space_extend:
            max_space_extend = space_extend
    return new_vertices * verts_range / (np.sqrt(3) * max_space_extend)

def adjust_mesh_diagonally(vertices, diag_range=1.0):
    device = vertices.get_device()
    min_coords = torch.min(vertices, dim=0)[0]
    max_coords = torch.max(vertices, dim=0)[0]
    diag = torch.linalg.norm(max_coords - min_coords)
    center = (min_coords + max_coords) / 2
    return (vertices - center) * diag_range / diag

def adjust_mesh_init(vertices):
    min_v = torch.min(vertices)
    max_v = torch.max(vertices)
    mean = torch.mean(vertices, axis=0)
    new_vertices = (vertices - min_v) / (max_v - min_v) - 0.5
    return new_vertices

def scale_mesh(mesh_verts, scale, offset):
    return mesh_verts * scale + offset

def random_scale_mesh(mesh_verts, min_scale, max_scale, output_max_range, input_range=1):
    scale = min_scale + (max_scale - min_scale) * torch.rand(1, device=mesh_verts.get_device())
    offset_factor = (output_max_range - scale * input_range) * 0.5
    offset = offset_factor * torch.rand(1, 3, device=mesh_verts.get_device())
    new_verts = scale_mesh(mesh_verts, scale, offset)
    return (new_verts, scale, offset)

def make_rasterizer(params, device, accurate=True, image_size=None, camera_dist=None, elevation=None, azim_angle=None):
    if params is not None:
        image_size = params.image_size
        camera_dist = params.camera_dist
        elevation = params.elevation
        azim_angle = params.azim_angle
    R, T = look_at_view_transform(dist=camera_dist, elev=elevation, azim=azim_angle)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    if accurate:
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=50000, perspective_correct=True)
    else:
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, perspective_correct=True)
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

def make_screen_rasterizer(params, n_screen_cameras, screen_cameras, device, accurate=False):
    if accurate:
        screen_raster_settings = RasterizationSettings(image_size=params.image_size, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=50000, perspective_correct=True)
    else:
        screen_raster_settings = RasterizationSettings(image_size=params.image_size, blur_radius=0.0, faces_per_pixel=1, perspective_correct=True)
    screen_rasterizer = MeshRasterizer(cameras=screen_cameras, raster_settings=screen_raster_settings)
    return screen_rasterizer

def make_screen_rgb_renderer(params, n_screen_cameras, screen_cameras, screen_lights, device, accurate=False):
    if accurate:
        screen_raster_settings = RasterizationSettings(image_size=params.image_size, blur_radius=0, faces_per_pixel=10, max_faces_per_bin=50000)
    else:
        screen_raster_settings = RasterizationSettings(image_size=params.image_size, blur_radius=0, faces_per_pixel=10)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=screen_cameras, raster_settings=screen_raster_settings), shader=SoftFlatShader(device=device, cameras=screen_cameras, lights=screen_lights))
    return renderer

def make_random_lights(n_screen_cameras, device, dist=1.5):
    elev = -90.0 + 180.0 * torch.rand(1, device=device)
    azim = 360.0 * torch.rand(1, device=device)
    light_ambiant_color = torch.rand(1, 3, device=device).expand(n_screen_cameras, -1)
    light_elev = elev.expand(n_screen_cameras).view(-1, 1)
    light_azim = azim.expand(n_screen_cameras).view(-1, 1)
    light_r = dist
    light_location = get_cartesian_coords(r=light_r, elev=light_elev, azim=light_azim, in_degrees=True)
    lights = PointLights(ambient_color=light_ambiant_color, device=device, location=light_location)
    return lights

class SoftFlatShader(nn.Module):

    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = materials if materials is not None else Materials(device=device)
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get('cameras', self.cameras)
        if cameras is None:
            msg = 'Cameras must be specified either at initialization                 or in the forward pass of HardFlatShader'
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get('lights', self.lights)
        materials = kwargs.get('materials', self.materials)
        blend_params = kwargs.get('blend_params', self.blend_params)
        colors = flat_shading(meshes=meshes, fragments=fragments, texels=texels, lights=lights, cameras=cameras, materials=materials)
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images

def load_mesh_from_path(model_path, device):
    model = model_path
    verts, faces, aux = load_obj(model, device=device, load_textures=False, create_texture_atlas=True, texture_atlas_size=4, texture_wrap='repeat')
    verts = adjust_mesh(verts)
    atlas = aux.texture_atlas
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=TexturesAtlas(atlas=[atlas]))
    return mesh

def compute_gt_surface(mesh, screen_rasterizer, n_points=-1, z_max=10.0):
    n_camera = 8
    device = mesh.device
    dist = [1.5 for i in range(n_camera)]
    elev = [45.0, 45.0, 45.0, 45.0, -45.0, -45.0, -45.0, -45.0]
    azim = [45.0, 90.0, 225.0, 315.0, 45.0, 90.0, 225.0, 315.0]
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    fragments = screen_rasterizer(mesh.extend(n_camera), cameras=cameras)
    surface_coords_per_pixel = compute_coords_per_pixel(mesh, fragments, device)
    surface_points = surface_coords_per_pixel[surface_coords_per_pixel[..., 2] < z_max]
    surface_points = surface_points[torch.randperm(len(surface_points))[:n_points]]
    return surface_points

def compute_dm_surface(mesh, cameras, screen_rasterizer, n_points=-1, z_max=10.0):
    n_camera = len(cameras)
    device = mesh.device
    fragments = screen_rasterizer(mesh.extend(n_camera), cameras=cameras)
    surface_coords_per_pixel = compute_coords_per_pixel(mesh, fragments, device)
    surface_points = surface_coords_per_pixel[surface_coords_per_pixel[..., 2] < z_max]
    surface_points = surface_points[torch.randperm(len(surface_points))[:n_points]]
    return surface_points

def compute_surface_epsilon(X_surf, quantile=0.9):
    distances = torch.cdist(X_surf, X_surf, p=2.0)
    distances[distances == 0.0] = 1000.0
    min_distances = torch.min(distances, dim=-1)[0]
    epsilon = torch.quantile(min_distances, quantile)
    return epsilon

def compute_coords_per_pixel(mesh, fragments, device, bg_value=10):
    nface = mesh.faces_list()[0]
    nverts = mesh.verts_list()[0]
    npixes = fragments.pix_to_face.squeeze(dim=3)
    nbary_coords = fragments.bary_coords.squeeze(dim=3)
    nb_faces = nface.shape[0]
    triangle_idx_per_pixel = npixes.long()
    triangle_vertices_per_pixel = nface[triangle_idx_per_pixel % nb_faces].long()
    bary_coords_per_pixel = nbary_coords
    xverts = nverts[..., 0]
    yverts = nverts[..., 1]
    zverts = nverts[..., 2]
    coord_per_pixel = torch.zeros(bary_coords_per_pixel.shape, device=device)
    coord_per_pixel[..., 0] = torch.sum(xverts[triangle_vertices_per_pixel] * bary_coords_per_pixel, axis=3)
    coord_per_pixel[..., 1] = torch.sum(yverts[triangle_vertices_per_pixel] * bary_coords_per_pixel, axis=3)
    coord_per_pixel[..., 2] = torch.sum(zverts[triangle_vertices_per_pixel] * bary_coords_per_pixel, axis=3)
    coord_per_pixel[bary_coords_per_pixel == -1] = bg_value
    return coord_per_pixel

def compute_coords_per_pixel_cpu(mesh, fragments):
    nface = mesh.faces_list()[0].cpu().numpy()
    nverts = mesh.verts_list()[0].cpu().numpy()
    npixes = fragments.pix_to_face.squeeze().cpu().numpy()
    nbary_coords = fragments.bary_coords.squeeze().cpu().numpy()
    nb_faces = nface.shape[0]
    triangle_idx_per_pixel = npixes
    triangle_vertices_per_pixel = nface[triangle_idx_per_pixel % nb_faces]
    bary_coords_per_pixel = nbary_coords
    xverts = nverts[..., 0]
    yverts = nverts[..., 1]
    zverts = nverts[..., 2]
    coord_per_pixel = np.zeros(bary_coords_per_pixel.shape)
    coord_per_pixel[..., 0] = np.sum(xverts[triangle_vertices_per_pixel] * bary_coords_per_pixel, axis=3)
    coord_per_pixel[..., 1] = np.sum(yverts[triangle_vertices_per_pixel] * bary_coords_per_pixel, axis=3)
    coord_per_pixel[..., 2] = np.sum(zverts[triangle_vertices_per_pixel] * bary_coords_per_pixel, axis=3)
    coord_per_pixel[bary_coords_per_pixel == -1] = 10
    return coord_per_pixel

def behind_depth_map3(X, coords_per_pixel_single, image_size, side, x_camera, y_camera, z_camera, device):
    tol = 1
    x_pixels = torch.flatten(coords_per_pixel_single[..., 0])
    y_pixels = torch.flatten(coords_per_pixel_single[..., 1])
    z_pixels = torch.flatten(coords_per_pixel_single[..., 2])
    x_camera_exp = torch.unsqueeze(x_camera, axis=1)
    y_camera_exp = torch.unsqueeze(y_camera, axis=1)
    z_camera_exp = torch.unsqueeze(z_camera, axis=1)
    x = torch.tensordot(X, x_camera_exp, dims=1)
    y = torch.tensordot(X, y_camera_exp, dims=1)
    sample_depth = torch.tensordot(X, z_camera_exp, dims=1)
    q_x = x // side
    q_y = y // side
    x_1 = q_x * side
    y_1 = q_y * side
    x_1 = torch.zeros_like(x, device=device)
    x_2 = torch.zeros_like(x, device=device)
    y_1 = torch.zeros_like(y, device=device)
    y_2 = torch.zeros_like(y, device=device)
    y_image_low = torch.zeros_like(x, device=device).long()
    y_image_high = torch.zeros_like(x, device=device).long()
    x_image_low = torch.zeros_like(y, device=device).long()
    x_image_high = torch.zeros_like(y, device=device).long()
    mask = x > (q_x + 0.5) * side
    x_1[mask] = (q_x[mask] + 1) * side
    x_2[mask] = q_x[mask] * side
    y_image_low[mask] = -1 * (q_x[mask].long() + 1) + image_size // 2
    y_image_high[mask] = -1 * q_x[mask].long() + image_size // 2
    mask = x <= (q_x + 0.5) * side
    x_1[mask] = q_x[mask] * side
    x_2[mask] = (q_x[mask] - 1) * side
    y_image_low[mask] = -1 * q_x[mask].long() + image_size // 2
    y_image_high[mask] = -1 * (q_x[mask].long() - 1) + image_size // 2
    mask = y > (q_y + 0.5) * side
    y_1[mask] = (q_y[mask] + 1) * side
    y_2[mask] = q_y[mask] * side
    x_image_low[mask] = 1 * (q_y[mask].long() + 1) + image_size // 2
    x_image_high[mask] = -1 * q_y[mask].long() + image_size // 2
    mask = y <= (q_y + 0.5) * side
    y_1[mask] = q_y[mask] * side
    y_2[mask] = (q_y[mask] - 1) * side
    x_image_low[mask] = -1 * q_y[mask].long() + image_size // 2
    x_image_high[mask] = -1 * (q_y[mask].long() - 1) + image_size // 2
    y_image_low[y_image_low < 0] = 0
    x_image_low[x_image_low < 0] = 0
    y_image_high[y_image_high >= image_size] = image_size - 1
    x_image_high[x_image_high >= image_size] = image_size - 1
    indices11 = image_size * x_image_low + y_image_low
    indices12 = image_size * x_image_high + y_image_low
    indices21 = image_size * x_image_low + y_image_high
    indices22 = image_size * x_image_high + y_image_high
    res11 = torch.stack((x_pixels[indices11], y_pixels[indices11], z_pixels[indices11]), axis=1)
    res11 = torch.squeeze(res11, axis=-1)
    res12 = torch.stack((x_pixels[indices12], y_pixels[indices12], z_pixels[indices12]), axis=1)
    res12 = torch.squeeze(res12, axis=-1)
    res21 = torch.stack((x_pixels[indices21], y_pixels[indices21], z_pixels[indices21]), axis=1)
    res21 = torch.squeeze(res21, axis=-1)
    res22 = torch.stack((x_pixels[indices22], y_pixels[indices22], z_pixels[indices22]), axis=1)
    res22 = torch.squeeze(res22, axis=-1)
    if True:
        res = 1 / (side * side) * (res11 * (x_2 - x) * (y_2 - y) + res12 * (x_2 - x) * (y - y_1) + res21 * (x - x_1) * (y_2 - y) + res22 * (x - x_1) * (y - y_1))
        bilinear_res_depth = torch.tensordot(res, z_camera_exp, dims=1)
    if True:
        res11_depth = torch.tensordot(res11, z_camera_exp, dims=1)
        res12_depth = torch.tensordot(res12, z_camera_exp, dims=1)
        res21_depth = torch.tensordot(res21, z_camera_exp, dims=1)
        res22_depth = torch.tensordot(res22, z_camera_exp, dims=1)
        res_depth = torch.cat((res11_depth, res12_depth, res21_depth, res22_depth), dim=-1)
    max_res_depth = torch.max(res_depth, dim=-1)[0]
    max_res_depth = torch.unsqueeze(max_res_depth, dim=-1)
    res_depth = torch.min(res_depth, dim=-1)[0]
    res_depth = torch.unsqueeze(res_depth, dim=-1)
    res_depth[max_res_depth < 5.0] = bilinear_res_depth[max_res_depth < 5.0]
    return res_depth - tol * np.sqrt(3) * side <= sample_depth

def behind_depth_map(X, coords_per_pixel_single, image_size, side, x_camera, y_camera, z_camera, device):
    tol = 1.0
    x_pixels = torch.flatten(coords_per_pixel_single[..., 0])
    y_pixels = torch.flatten(coords_per_pixel_single[..., 1])
    z_pixels = torch.flatten(coords_per_pixel_single[..., 2])
    x_camera_exp = torch.unsqueeze(x_camera, axis=1)
    y_camera_exp = torch.unsqueeze(y_camera, axis=1)
    z_camera_exp = torch.unsqueeze(z_camera, axis=1)
    x = torch.tensordot(X, x_camera_exp, dims=1)
    y = torch.tensordot(X, y_camera_exp, dims=1)
    sample_depth = torch.tensordot(X, z_camera_exp, dims=1)
    q_x = floor_divide(x, side)
    q_y = floor_divide(y, side)
    x_1 = q_x * side
    y_1 = q_y * side
    y_image_1 = -1 * q_x.long() + image_size // 2
    x_image_1 = -1 * q_y.long() + image_size // 2
    y_image_2 = y_image_1 - 1
    x_image_2 = x_image_1 - 1
    y_image_0 = y_image_1 + 1
    x_image_0 = x_image_1 + 1
    y_image_2[y_image_2 < 0] = 0
    x_image_2[x_image_2 < 0] = 0
    y_image_0[y_image_2 >= image_size] = image_size - 1
    x_image_0[x_image_2 >= image_size] = image_size - 1
    indices11 = image_size * x_image_1 + y_image_1
    indices12 = image_size * x_image_2 + y_image_1
    indices21 = image_size * x_image_1 + y_image_2
    indices22 = image_size * x_image_2 + y_image_2
    res11 = torch.stack((x_pixels[indices11], y_pixels[indices11], z_pixels[indices11]), axis=1)
    res11 = torch.squeeze(res11, axis=-1)
    res12 = torch.stack((x_pixels[indices12], y_pixels[indices12], z_pixels[indices12]), axis=1)
    res12 = torch.squeeze(res12, axis=-1)
    res21 = torch.stack((x_pixels[indices21], y_pixels[indices21], z_pixels[indices21]), axis=1)
    res21 = torch.squeeze(res21, axis=-1)
    res22 = torch.stack((x_pixels[indices22], y_pixels[indices22], z_pixels[indices22]), axis=1)
    res22 = torch.squeeze(res22, axis=-1)
    indices00 = image_size * x_image_0 + y_image_0
    indices10 = image_size * x_image_0 + y_image_1
    indices01 = image_size * x_image_1 + y_image_0
    indices20 = image_size * x_image_0 + y_image_2
    indices02 = image_size * x_image_2 + y_image_0
    res00 = torch.stack((x_pixels[indices00], y_pixels[indices00], z_pixels[indices00]), axis=1)
    res00 = torch.squeeze(res00, axis=-1)
    res10 = torch.stack((x_pixels[indices10], y_pixels[indices10], z_pixels[indices10]), axis=1)
    res10 = torch.squeeze(res10, axis=-1)
    res01 = torch.stack((x_pixels[indices01], y_pixels[indices01], z_pixels[indices01]), axis=1)
    res01 = torch.squeeze(res01, axis=-1)
    res20 = torch.stack((x_pixels[indices20], y_pixels[indices20], z_pixels[indices20]), axis=1)
    res20 = torch.squeeze(res20, axis=-1)
    res02 = torch.stack((x_pixels[indices02], y_pixels[indices02], z_pixels[indices02]), axis=1)
    res02 = torch.squeeze(res02, axis=-1)
    res11_depth = torch.tensordot(res11, z_camera_exp, dims=1)
    res12_depth = torch.tensordot(res12, z_camera_exp, dims=1)
    res21_depth = torch.tensordot(res21, z_camera_exp, dims=1)
    res22_depth = torch.tensordot(res22, z_camera_exp, dims=1)
    res00_depth = torch.tensordot(res00, z_camera_exp, dims=1)
    res10_depth = torch.tensordot(res10, z_camera_exp, dims=1)
    res01_depth = torch.tensordot(res01, z_camera_exp, dims=1)
    res20_depth = torch.tensordot(res20, z_camera_exp, dims=1)
    res02_depth = torch.tensordot(res02, z_camera_exp, dims=1)
    res_depth = torch.cat((res11_depth, res12_depth, res21_depth, res22_depth, res00_depth, res10_depth, res01_depth, res20_depth, res02_depth), dim=-1)
    res_depth = torch.min(res_depth, dim=-1)[0]
    res_depth = torch.unsqueeze(res_depth, dim=-1)
    return res_depth - tol * np.sqrt(3) * side <= sample_depth

def behind_depth_map2(X, coords_per_pixel_single, image_size, side, x_camera, y_camera, z_camera, device):
    tol = 1
    x_pixels = torch.flatten(coords_per_pixel_single[..., 0])
    y_pixels = torch.flatten(coords_per_pixel_single[..., 1])
    z_pixels = torch.flatten(coords_per_pixel_single[..., 2])
    x_camera_exp = torch.unsqueeze(x_camera, axis=1)
    y_camera_exp = torch.unsqueeze(y_camera, axis=1)
    z_camera_exp = torch.unsqueeze(z_camera, axis=1)
    y_image = -1 * (torch.tensordot(X, x_camera_exp, dims=1) // side).long() + image_size // 2
    x_image = -1 * (torch.tensordot(X, y_camera_exp, dims=1) // side).long() + image_size // 2
    sample_depth = torch.tensordot(X, z_camera_exp, dims=1)
    indices = image_size * x_image + y_image
    res = torch.stack((x_pixels[indices], y_pixels[indices], z_pixels[indices]), axis=1)
    res = torch.squeeze(res, axis=-1)
    return torch.tensordot(res, z_camera_exp, dims=1) - tol * np.sqrt(3) * side <= sample_depth

def behind_depth_map_cpu(X, coords_per_pixel_single, image_size, side, x_camera, y_camera, z_camera):
    tol = 1
    x_pixels = coords_per_pixel_single[..., 0].flatten()
    y_pixels = coords_per_pixel_single[..., 1].flatten()
    z_pixels = coords_per_pixel_single[..., 2].flatten()
    y_image = -1 * (np.dot(X, x_camera) // side).astype(int) + image_size // 2
    x_image = -1 * (np.dot(X, y_camera) // side).astype(int) + image_size // 2
    sample_depth = np.dot(X, z_camera)
    indices = image_size * x_image + y_image
    res = np.stack((x_pixels[indices], y_pixels[indices], z_pixels[indices]), axis=1)
    return np.dot(res, z_camera) - tol * np.sqrt(3) * side <= sample_depth

def compute_occupancy(X, coords_per_pixel, image_size, side, camera_axes, device, resolution_factor=1.0, edge_tol=5):
    res = torch.ones((len(X), 1), device=device).bool()
    if resolution_factor > 1.0:
        X_cpt = resolution_factor * X
        X_cpt[X_cpt < side * (-image_size / 2 + edge_tol)] = side * (-image_size / 2 + edge_tol)
        X_cpt[X_cpt > side * (-image_size / 2 + image_size - edge_tol)] = side * (-image_size / 2 + image_size - edge_tol)
    else:
        X_cpt = X
    for i in range(len(coords_per_pixel)):
        x_camera, y_camera, z_camera = (camera_axes[i, 0], camera_axes[i, 1], camera_axes[i, 2])
        res *= behind_depth_map(X_cpt, coords_per_pixel[i], image_size, side, x_camera, y_camera, z_camera, device)
    res = torch.squeeze(res, axis=-1)
    return res

def compute_occupancy_cpu(X, coords_per_pixel, image_size, side, camera_axes):
    res = np.ones(len(X)).astype(bool)
    for i in range(len(coords_per_pixel)):
        x_camera, y_camera, z_camera = (camera_axes[i, 0], camera_axes[i, 1], camera_axes[i, 2])
        res *= behind_depth_map_cpu(X, coords_per_pixel[i], image_size, side, x_camera, y_camera, z_camera)
    return res

def sample_X_in_box(x_range, n_sample, device):
    return -x_range / 2.0 + x_range * torch.rand(n_sample, 3, device=device)

def sample_X_in_ball(x_radius, n_sample, device):
    units = torch.randn((n_sample, 3), device=device)
    units /= torch.linalg.norm(units, axis=1, keepdim=True)
    norms = x_radius * torch.sqrt(torch.rand((n_sample, 1), device=device))
    return norms * units

def sample_X_in_ball2(x_radius, n_sample, device):
    alphas = 2 * np.pi * torch.rand((n_sample, 1), device=device)
    betas = np.pi * (-1 / 2 + torch.rand((n_sample, 1), device=device))
    norms = x_radius * torch.rand((n_sample, 1), device=device)
    return torch.hstack((torch.cos(betas) * torch.sin(alphas), torch.sin(betas), torch.cos(betas) * torch.cos(alphas)))

def compute_mesh_face_area_2(verts, faces):
    face_coords = verts[faces]
    a = torch.linalg.norm(face_coords[..., 0, :] - face_coords[..., 1, :], dim=-1)
    b = torch.linalg.norm(face_coords[..., 1, :] - face_coords[..., 2, :], dim=-1)
    c = torch.linalg.norm(face_coords[..., 2, :] - face_coords[..., 0, :], dim=-1)
    p = (a + b + c) / 2.0
    res = p * (p - a) * (p - b) * (p - c)
    res[res < 0.0] = 0.0
    return torch.sqrt(res)

def compute_mesh_face_area(verts, faces):
    face_coords = verts[faces]
    a = torch.linalg.norm(face_coords[..., 0, :] - face_coords[..., 1, :], dim=-1)
    b = torch.linalg.norm(face_coords[..., 1, :] - face_coords[..., 2, :], dim=-1)
    c = torch.linalg.norm(face_coords[..., 2, :] - face_coords[..., 0, :], dim=-1)
    p = (a + b + c) / 2.0
    precision = len(p[p <= 0]) == 0
    f1 = p
    f2 = p - a
    f3 = p - b
    f4 = p - c
    if precision:
        f1 = 1.0
        f2 = f2 / p
        f3 = f3 / p
        f4 = f4 / p
    res = f1 * f2 * f3 * f4
    res[res < 0.0] = 0.0
    res = torch.sqrt(res)
    if precision:
        res *= p * p
    return res

def sample_mesh_triangle(verts, faces, n_sample, threshold=10000000.0):
    n_faces = faces.shape[0]
    device = verts.get_device()
    n_sample_max = int(threshold / n_faces)
    if n_sample_max == 0:
        raise NameError('The mesh has too many faces and would cause memory issues. Please use lighter meshes.')
    q = n_sample // n_sample_max
    n_iter = q
    if n_sample % n_sample_max > 0:
        n_iter += 1
    sample_probs = compute_mesh_face_area(verts, faces)
    sample_probs /= torch.sum(sample_probs)
    sample_probs = torch.cumsum(sample_probs, dim=-1)
    res = torch.zeros(0, device=device).long()
    for i in range(n_iter):
        if i == n_iter - 1 and n_sample % n_sample_max > 0:
            n_sample_i = n_sample % n_sample_max
        else:
            n_sample_i = n_sample_max
        samples = torch.rand(n_sample_i, 1, device=device)
        res_i = sample_probs.view(1, n_faces).expand(n_sample_i, -1) - samples.expand(-1, n_faces)
        res_i[res_i < 0] = 2
        res_i = torch.argmin(res_i, dim=-1)
        res = torch.cat((res, res_i), dim=0)
    return res

def sample_mesh_triangle_simple(verts, faces, n_sample):
    n_faces = faces.shape[0]
    device = verts.get_device()
    sample_probs = compute_mesh_face_area(verts, faces)
    sample_probs /= torch.sum(sample_probs)
    sample_probs = torch.cumsum(sample_probs, dim=-1)
    samples = torch.rand(n_sample, 1, device=device)
    res = sample_probs.view(1, n_faces).expand(n_sample, -1) - samples.expand(-1, n_faces)
    res[res < 0] = 2
    res = torch.argmin(res, dim=-1)
    return res

def sample_points_on_mesh_faces(verts, faces, sample_face_indices, return_textures=False, mesh=None, texture_face_indices=None):
    device = verts.get_device()
    n_sample = sample_face_indices.shape[0]
    sample_faces = faces[sample_face_indices]
    face_coords = verts[sample_faces]
    o = face_coords[..., 2, :]
    a = face_coords[..., 0, :] - o
    b = face_coords[..., 1, :] - o
    alpha = torch.rand(n_sample, 1, device=device).expand(-1, 3).clone()
    beta = torch.rand(n_sample, 1, device=device).expand(-1, 3).clone()
    mask = alpha + beta > 1.0
    alpha[mask] = 1.0 - alpha[mask]
    beta[mask] = 1.0 - beta[mask]
    if return_textures:
        pix_to_face = texture_face_indices.view(1, -1, 1, 1)
        bary = torch.stack((alpha[..., 0:1], beta[..., 0:1], (1 - (alpha + beta))[..., 0:1]), dim=1).view(1, -1, 1, 1, 3)
        dummy = torch.zeros(1, n_sample, 1, 1, device=device)
        fragments = MeshFragments(pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy)
        textures = mesh.sample_textures(fragments)
        textures = textures[0, :, 0, 0, :]
        return (o + alpha * a + beta * b, textures)
    return o + alpha * a + beta * b

def sample_points_on_mesh_surface(verts, faces, n_sample, return_colors=False, mesh=None, texture_face_indices=None):
    sample_face_indices = sample_mesh_triangle(verts, faces, n_sample)
    if texture_face_indices is not None:
        texture_face_indices = texture_face_indices[sample_face_indices]
    res = sample_points_on_mesh_faces(verts, faces, sample_face_indices, return_textures=return_colors, mesh=mesh, texture_face_indices=texture_face_indices)
    return res

def project_depth_back_to_3D(depth, cameras):
    n_cameras, image_height, image_width = (depth.shape[0], depth.shape[1], depth.shape[2])
    device = depth.get_device()
    if device < 0:
        device = 'cpu'
    x_tab = torch.Tensor([[i for j in range(image_width)] for i in range(image_height)]).to(device)
    y_tab = torch.Tensor([[j for j in range(image_width)] for i in range(image_height)]).to(device)
    ndc_x_tab = image_width / min(image_width, image_height) - y_tab / (min(image_width, image_height) - 1) * 2
    ndc_y_tab = image_height / min(image_width, image_height) - x_tab / (min(image_width, image_height) - 1) * 2
    ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(n_cameras, -1, -1), ndc_y_tab.view(1, -1, 1).expand(n_cameras, -1, -1), depth.view(n_cameras, -1, 1)), dim=-1).view(n_cameras, image_height * image_width, 3)
    all_world_points = cameras.unproject_points(ndc_points, scaled_depth_input=False)
    points_mask = (depth > -1).view(n_cameras, -1)
    return all_world_points[points_mask]
'\nCBN (Conditional Batch Normalization layer)\n    uses an MLP to predict the beta and gamma parameters in the batch norm equation\n    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf\n'

def get_knn_points(X, pc, k):
    dists = torch.cdist(X, pc)
    min_dists, argmin_dists = torch.topk(dists, k=k, dim=-1, largest=False)
    return (knn_gather(pc, argmin_dists), min_dists, argmin_dists)

def get_k_nearest_ray_points(X_camera, X, pc, k):
    X_cam = X_camera.view(-1, 1, 3)
    rays = pc - X_cam
    rays /= torch.linalg.norm(rays)
    x_rays = X - X_cam
    x_rays /= torch.linalg.norm(x_rays)
    dots = x_rays.matmul(rays.transpose(dim0=-2, dim1=-1))
    max_dots, argmax_dots = torch.topk(dots, k=k, dim=-1)
    return (knn_gather(pc, argmax_dots), max_dots, argmax_dots)

class CBN2(nn.Module):

    def __init__(self, lstm_size, emb_size, out_size, batch_size, channels, height, width, use_betas=True, use_gammas=True, eps=1e-05):
        super(CBN2, self).__init__()
        self.lstm_size = lstm_size
        self.emb_size = emb_size
        self.out_size = out_size
        self.use_betas = use_betas
        self.use_gammas = use_gammas
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps
        self.fc_gamma = nn.Sequential(nn.Linear(self.lstm_size, self.emb_size), nn.ReLU(inplace=True), nn.Linear(self.emb_size, self.out_size)).cuda()
        self.fc_beta = nn.Sequential(nn.Linear(self.lstm_size, self.emb_size), nn.ReLU(inplace=True), nn.Linear(self.emb_size, self.out_size)).cuda()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)
    '\n    Predicts the value of delta beta and delta gamma for each channel\n    Arguments:\n        lstm_emb : lstm embedding of the question\n    Returns:\n        delta_betas, delta_gammas : for each layer\n    '

    def create_cbn_input(self, lstm_emb):
        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()
        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()
        return (delta_betas, delta_gammas)
    '\n    Computer Normalized feature map with the updated beta and gamma values\n    Arguments:\n        feature : feature map from the previous layer\n        lstm_emb : lstm embedding of the question\n    Returns:\n        out : beta and gamma normalized feature map\n        lstm_emb : lstm embedding of the question (unchanged)\n    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential\n           and subsequent CBN layers will also require lstm question embeddings\n    '

    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)
        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)
        betas_expanded = torch.stack([betas_cloned] * self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded] * self.width, dim=3)
        gammas_expanded = torch.stack([gammas_cloned] * self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded] * self.width, dim=3)
        feature_normalized = (feature - batch_mean) / torch.sqrt(batch_var + self.eps)
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded
        return (out, lstm_emb)
