import io
import torch
import ast
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import trimesh

def generate_hash_table(image):
    hash_table = {}
    indices = image.nonzero()
    for index in indices:
        hash_table[tuple(index.tolist())] = True
    return hash_table

def dda_line(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    x_inc = dx / steps
    y_inc = dy / steps
    x = x0
    y = y0
    points_on_line = []
    for _ in range(int(steps) + 1):
        points_on_line.append((int(round(x)), int(round(y))))
        x += x_inc
        y += y_inc
    return points_on_line

def check_line_intersection(pc, point1, point2):
    image, x_min, z_min, scale_x, scale_z, image_size = generate_image_and_scale_factors(pc)
    hash_table = generate_hash_table(image)
    x0 = int((point1[0] - x_min) * scale_x)
    z0 = int((point1[2] - z_min) * scale_z)
    x1 = int((point2[0] - x_min) * scale_x)
    z1 = int((point2[2] - z_min) * scale_z)
    for x, z in dda_line(x0, z0, x1, z1):
        if (z, x) in hash_table:
            return True
    return False

def check_line_no_collision(pc, point1, point2):
    image, x_min, z_min, scale_x, scale_z, image_size = generate_image_and_scale_factors(pc)
    hash_table = generate_hash_table(image)
    x0 = int((point1[0] - x_min) * scale_x)
    z0 = int((point1[2] - z_min) * scale_z)
    x1 = int((point2[0] - x_min) * scale_x)
    z1 = int((point2[2] - z_min) * scale_z)
    if (z0, x0) in hash_table and (z1, x1) in hash_table:
        return True
    return False

def generate_image_and_scale_factors(pc):
    x_min, x_max = (torch.min(pc[:, 0]), torch.max(pc[:, 0]))
    z_min, z_max = (torch.min(pc[:, 2]), torch.max(pc[:, 2]))
    image_size = max(int(x_max - x_min) + 1, int(z_max - z_min) + 1)
    image = torch.zeros((image_size, image_size), dtype=torch.bool)
    scale_x = (image_size - 1) / (x_max - x_min)
    scale_z = (image_size - 1) / (z_max - z_min)
    x_indices = ((pc[:, 0] - x_min) * scale_x).round().long()
    z_indices = ((pc[:, 2] - z_min) * scale_z).round().long()
    image[z_indices, x_indices] = True
    return (image, x_min, z_min, scale_x, scale_z, image_size)

def calculate_similarity_pcs(gt_scene_pc, full_pc):
    x_min, x_max = (gt_scene_pc[:, 0].min(), gt_scene_pc[:, 0].max())
    z_min, z_max = (gt_scene_pc[:, 2].min(), gt_scene_pc[:, 2].max())
    mask = (full_pc[:, 0] >= x_min) & (full_pc[:, 0] <= x_max) & (full_pc[:, 2] >= z_min) & (full_pc[:, 2] <= z_max)
    filtered_full_pc = full_pc[mask]
    image_width = int((x_max - x_min) * 5)
    image_height = int((z_max - z_min) * 5)

    def generate_image(pc):
        image = torch.zeros((image_height, image_width), dtype=torch.uint8)
        x = ((pc[:, 0] - x_min) / (x_max - x_min) * (image_width - 1)).round().long()
        z = ((pc[:, 2] - z_min) / (z_max - z_min) * (image_height - 1)).round().long()
        image[z, x] = 1
        return image
    img1 = generate_image(gt_scene_pc)
    img2 = generate_image(filtered_full_pc)
    true_positive = (img1 & img2).sum().item()
    total_relevant = img1.sum().item()
    similarity = true_positive / total_relevant if total_relevant else 0
    return similarity

def get_point_position_in_the_img(points_2d, grid_size, grid_range):
    scale_x, scale_y = (grid_size[0] / (grid_range[1] - grid_range[0]), grid_size[1] / (grid_range[1] - grid_range[0]))
    x_mapped = ((points_2d[..., 0] - grid_range[0]) * scale_x).round().long()
    y_mapped = ((points_2d[..., 1] - grid_range[0]) * scale_y).round().long()
    return torch.stack((x_mapped, y_mapped)).squeeze()

def transform_points_to_n_pieces(points, camera_pose, device, no_rotation=True):
    x, y, z, elevation, azimuth = camera_pose.tolist()
    if no_rotation:
        azimuth = 0
    azimuth = torch.tensor([azimuth], device=device)
    radians = azimuth * torch.pi / 180.0
    cos_a = torch.cos(radians)
    sin_a = torch.sin(radians)
    zero = torch.zeros_like(radians)
    one = torch.ones_like(radians)
    R = torch.stack([torch.stack([cos_a, zero, -sin_a], dim=-1), torch.stack([zero, one, zero], dim=-1), torch.stack([sin_a, zero, cos_a], dim=-1)], dim=-1).squeeze(0)
    t = torch.tensor([x, y, z], dtype=torch.float32, device=device).view(3, 1)
    R_transpose = R.t()
    inverse_t = -torch.matmul(R_transpose, t)
    cP = torch.matmul(R_transpose, points.t()) + inverse_t
    cP_grid = -cP[[2, 0], :].t().unsqueeze(0)
    return cP_grid

def map_points_to_n_imgs(points_2d_batch, grid_size, grid_range, device):
    n, m, _ = points_2d_batch.shape
    output = torch.zeros((n,) + grid_size, dtype=torch.float32, device=device)
    scale_x, scale_y = (grid_size[0] / (grid_range[1] - grid_range[0]), grid_size[1] / (grid_range[1] - grid_range[0]))
    x_mapped = ((points_2d_batch[..., 0] - grid_range[0]) * scale_x).round().long()
    y_mapped = ((points_2d_batch[..., 1] - grid_range[0]) * scale_y).round().long()
    valid_mask = (x_mapped >= 0) & (x_mapped < grid_size[0]) & (y_mapped >= 0) & (y_mapped < grid_size[1])
    valid_x = x_mapped[valid_mask]
    valid_y = y_mapped[valid_mask]
    indices = (torch.arange(n, device=points_2d_batch.device).view(n, 1).expand(n, m)[valid_mask.view(n, m)].view(-1), valid_x, valid_y)
    output.index_put_(indices, torch.ones_like(valid_x, dtype=torch.float32, device=device), accumulate=True)
    return output

def get_binary_obstacle_array(mesh, camera_pose, view_size=80):
    x, y_value, z, elevation, azimuth = camera_pose.tolist()
    center_point = [x, z]
    plane_origin = [0, y_value, 0]
    plane_normal = [0, 1, 0]
    intersection = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin)
    if intersection.size != 0:
        path = trimesh.load_path(intersection)
        fig, ax = plt.subplots(figsize=(2.56, 2.56))
        for entity in path.entities:
            points = path.vertices[entity.points]
            ax.plot(points[:, 0], points[:, 2], 'k')
        ax.axis('off')
        ax.set_aspect('equal', adjustable='datalim')
        ax.autoscale_view()
        ax.set_xlim([center_point[0] - view_size / 2, center_point[0] + view_size / 2])
        ax.set_ylim([center_point[1] - view_size / 2, center_point[1] + view_size / 2])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img = img.convert('L')
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        binary_img = img.point(lambda x: 0 if x > 128 else 1, '1')
        binary_array = np.array(binary_img, dtype=int)
        return binary_array

def batch_transform_points_optimized_poster(points, camera_pose, device, occupancy_values, num_splits=4):
    x, y, z, elevation, azimuth = camera_pose.tolist()
    azimuth = torch.tensor([azimuth], device=device)
    radians = azimuth * torch.pi / 180.0
    cos_a = torch.cos(radians)
    sin_a = torch.sin(radians)
    zero = torch.zeros_like(radians)
    one = torch.ones_like(radians)
    R = torch.stack([torch.stack([cos_a, zero, -sin_a], dim=-1), torch.stack([zero, one, zero], dim=-1), torch.stack([sin_a, zero, cos_a], dim=-1)], dim=-1).squeeze(0)
    t = torch.tensor([x, y, z], dtype=torch.float32, device=device).view(3, 1)
    R_transpose = R.t()
    inverse_t = -torch.matmul(R_transpose, t)
    y_min, y_max = (points[:, 1].min().item(), points[:, 1].max().item())
    y_step = (y_max - y_min) / num_splits
    chunks = []
    for i in range(num_splits):
        y_lower = y_min + i * y_step
        y_upper = y_min + (i + 1) * y_step
        if i == num_splits - 1:
            mask = (points[:, 1] >= y_lower) & (points[:, 1] <= y_upper)
        else:
            mask = (points[:, 1] >= y_lower) & (points[:, 1] < y_upper)
        chunk_points = points[mask]
        chunk_occupancy = occupancy_values[mask]
        chunks.append((chunk_points, chunk_occupancy))
    transformed_chunks = []
    for chunk_points, chunk_occupancy in chunks:
        if chunk_points.shape[0] == 0:
            continue
        cP = torch.matmul(R_transpose, chunk_points.t()) + inverse_t
        cP_grid = -cP[[2, 0], :].t().unsqueeze(0)
        transformed_chunks.append((cP_grid, chunk_occupancy))
    return transformed_chunks

def map_points_to_grid_optimized_poster(points_2d_batch, occupancy_values, grid_size, grid_range):
    n, m, _ = points_2d_batch.shape
    device = occupancy_values.device
    output = torch.zeros((n,) + grid_size, dtype=torch.float32, device=device)
    scale_x, scale_y = (grid_size[0] / (grid_range[1] - grid_range[0]), grid_size[1] / (grid_range[1] - grid_range[0]))
    x_mapped = ((points_2d_batch[..., 0] - grid_range[0]) * scale_x).round().long()
    y_mapped = ((points_2d_batch[..., 1] - grid_range[0]) * scale_y).round().long()
    valid_mask = (x_mapped >= 0) & (x_mapped < grid_size[0]) & (y_mapped >= 0) & (y_mapped < grid_size[1])
    valid_x = x_mapped[valid_mask]
    valid_y = y_mapped[valid_mask]
    valid_mask_flat = valid_mask.view(-1)
    valid_occupancy_values = occupancy_values[valid_mask_flat]
    indices = (torch.arange(n, device=points_2d_batch.device).view(n, 1).expand(n, m)[valid_mask.view(n, m)].view(-1), valid_x, valid_y)
    valid_occupancy_values = valid_occupancy_values.squeeze(-1)
    output.index_put_(indices, valid_occupancy_values, accumulate=True)
    count_grid = torch.zeros_like(output)
    count_grid.index_put_(indices, torch.ones_like(valid_occupancy_values, device=device), accumulate=True)
    output[count_grid > 0] /= count_grid[count_grid > 0]
    return output

def plot_8_channel_heatmap(gain_map_prediction, current_camera_grids_img, pose_i, save_path):
    channels_to_plot = [gain_map_prediction[:, 2, :, :], gain_map_prediction[:, 1, :, :], gain_map_prediction[:, 0, :, :], gain_map_prediction[:, 3, :, :], None, gain_map_prediction[:, 0, :, :], gain_map_prediction[:, 5, :, :], gain_map_prediction[:, 6, :, :], gain_map_prediction[:, 7, :, :]]
    vmin = np.min([gain_map_prediction[:, i, :, :].cpu().min().item() for i in [0, 1, 2, 3, 5, 6, 7]])
    vmax = np.max([gain_map_prediction[:, i, :, :].cpu().max().item() for i in [0, 1, 2, 3, 5, 6, 7]])
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})
    for i, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        if channels_to_plot[i] is None:
            channel_image = current_camera_grids_img.cpu().numpy()
            im = ax.imshow(channel_image, cmap='magma')
        else:
            channel_image = channels_to_plot[i].squeeze().detach().cpu().numpy()
            im = ax.imshow(channel_image, cmap='nipy_spectral', vmin=vmin, vmax=vmax)
        ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'{save_path}/heatmap_prediction_{pose_i}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.clf()

def plot_full_pc(full_pc, pose_i, save_path):
    x = -full_pc[:, 0].cpu().numpy()
    z = full_pc[:, 2].cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(x, z, s=1, color='black')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{save_path}/results_{pose_i}.png')
    plt.clf()
