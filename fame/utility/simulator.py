import ast
import math
from pathlib import Path
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from pytorch3d.renderer import FoVPerspectiveCameras
from macarons.utility.CustomDataset import SceneDataset
from macarons.utility.macarons_utils import Camera, Settings, get_rgb_renderer, load_scene_with_texture, line_segment_mesh_intersection
from macarons.testers.scene import setup_test_scene
from .utils import transform_points_to_n_pieces, map_points_to_n_imgs

class MappingCamera(Camera):

    def get_fov_camera_from_RT(self, R_cam, T_cam):
        return FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=self.zfar, fov=90, device=self.device)

    def get_camera_parameters_from_pose(self, cam_pose):
        x, v, camera = super().get_camera_parameters_from_pose(cam_pose)
        return (x, v, self.get_fov_camera_from_RT(camera.R, camera.T))

    def update_camera(self, new_cam_index, interpolation_step=None):
        super().update_camera(new_cam_index, interpolation_step)
        self.fov_camera = self.get_fov_camera_from_RT(self.fov_camera.R, self.fov_camera.T)

class MappingScene:

    def __init__(self, params, cfg, scene_name, device):
        self.params, self.cfg, self.device = (params, cfg, torch.device(device))
        self.scale = float(params.scene_scale_factor)
        self.name = scene_name
        self.record = SceneDataset(params.data_path, [scene_name], use_occupied_pose=False)[0]
        self.settings = Settings(self.record['settings'], self.device, self.scale)
        path = Path(params.data_path) / scene_name / self.record['obj_name']
        self.mesh = load_scene_with_texture(str(path), self.scale, self.device)
        self.collision_mesh = trimesh.load(str(path), force='mesh')
        self.collision_mesh.vertices *= self.scale
        scenes = setup_test_scene(params, self.mesh, self.settings, False, self.device, surface_scene_feature_dim=3, test_resolution=0.05)
        self.gt_points = scenes[0].return_entire_pt_cloud(return_features=False)
        del scenes
        self.gt_numpy = self.gt_points.detach().cpu().numpy() / self.scale
        verts = self.mesh.verts_list()[0]
        low, high = (verts[:, 1].min(), verts[:, 1].max())
        self.y_bins = torch.linspace(low.item(), high.item(), 5, device=self.device)
        initial = FoVPerspectiveCameras(zfar=params.zfar, fov=90, device=self.device)
        renderer = get_rgb_renderer(image_height=params.image_height, image_width=params.image_width, ambient_light_intensity=params.ambient_light_intensity, cameras=initial, device=self.device, max_faces_per_bin=500000)
        s = self.settings.camera
        self.camera = MappingCamera(x_min=s.x_min, x_max=s.x_max, pose_l=s.pose_l, pose_w=s.pose_w, pose_h=s.pose_h, pose_n_elev=s.pose_n_elev, pose_n_azim=8, n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar, renderer=renderer, device=self.device, contrast_factor=s.contrast_factor, gathering_factor=params.gathering_factor, save_dir_path=None)
        self.starts = s.start_positions
        self.points = torch.empty(0, 3, device=self.device)
        self.colors = torch.empty(0, 3, device=self.device)
        self.trajectory = []
        self.collisions = 0

    def reset(self, start_number):
        index = self.starts[start_number % len(self.starts)].to(self.device).long()
        self.camera.cam_idx_history = torch.empty(0, 5, device=self.device)
        self.camera.X_cam_history = torch.empty(0, 3, device=self.device)
        self.camera.V_cam_history = torch.empty(0, 2, device=self.device)
        for key in self.camera.pose_history:
            self.camera.pose_history[key] = False
        self.camera.initialize_camera(index)
        pose, _ = self.camera.get_pose_from_idx(index)
        self.pose = pose.clone()
        self.points = torch.empty(0, 3, device=self.device)
        self.colors = torch.empty(0, 3, device=self.device)
        self.trajectory = [self.pose.clone()]
        self.collisions = 0
        self.observe()

    @torch.no_grad()
    def render(self, pose):
        _, _, camera = self.camera.get_camera_parameters_from_pose(pose)
        images, depth = self.camera.capture_image(self.mesh, fov_camera=camera, save_frame=False)
        points, colors = self.camera.compute_partial_point_cloud(depth=depth, mask=depth > 0, images=images, fov_cameras=camera, gathering_factor=self.params.gathering_factor, fov_range=self.params.sensor_range)
        return (points, colors)

    def observe(self):
        points, colors = self.render(self.pose)
        self.points = torch.cat([self.points, points])
        self.colors = torch.cat([self.colors, colors])
        if self.cfg.max_points and len(self.points) > self.cfg.max_points:
            indices = torch.randperm(len(self.points), device=self.device)[:self.cfg.max_points]
            self.points, self.colors = (self.points[indices], self.colors[indices])

    def state(self):
        n = self.cfg.image_size
        half = self.cfg.extent_m * self.scale / 2
        bounds = (-half, half)
        bins = torch.bucketize(self.points[:, 1], self.y_bins[1:-1])
        layers = []
        for layer in range(4):
            points = self.points[bins == layer]
            coords = transform_points_to_n_pieces(points, self.pose, self.device)
            layers.append(map_points_to_n_imgs(coords, (n, n), bounds, self.device))
        history = torch.stack(self.trajectory)[:, :3]
        projected = transform_points_to_n_pieces(history, self.pose, self.device)
        layers.append(map_points_to_n_imgs(projected, (n, n), bounds, self.device))
        return torch.cat(layers, 0).unsqueeze(0)

    def local_to_pose(self, point, origin=None):
        origin = self.pose if origin is None else origin
        pose = origin.clone()
        pose[0] -= float(point[0]) * self.scale
        pose[2] -= float(point[1]) * self.scale
        pose[3] = 0
        pose[4] = float(np.degrees(np.arctan2(-np.cos(point[2]), -np.sin(point[2]))) % 360)
        return pose

    def poses_to_local(self, poses, origin=None):
        origin = self.pose if origin is None else origin
        if not len(poses):
            return None
        values = torch.stack(poses).detach().cpu().numpy()
        center = origin.detach().cpu().numpy()
        xy = -(values[:, [0, 2]] - center[[0, 2]]) / self.scale
        azimuth = np.radians(values[:, 4])
        theta = np.arctan2(-np.cos(azimuth), -np.sin(azimuth))
        return np.column_stack([xy, theta])

    def path_poses(self, path, origin=None):
        return [self.local_to_pose(point, origin) for point in path]

    def blocked(self, a, b):
        if float(torch.linalg.vector_norm(a[:3] - b[:3])) < 1e-08:
            return False
        return bool(line_segment_mesh_intersection(a[:3], b[:3], self.collision_mesh))

    def execute(self, target):
        if self.blocked(self.pose, target):
            self.collisions += 1
            return False
        old = self.pose.clone()
        yaw_delta = (target[4] - old[4] + 180) % 360 - 180
        for step in range(1, self.params.n_interpolation_steps + 1):
            fraction = step / self.params.n_interpolation_steps
            self.pose = old + fraction * (target - old)
            self.pose[4] = old[4] + fraction * yaw_delta
            self.observe()
        self.pose = target.clone()
        self.trajectory.append(self.pose.clone())
        self.camera.X_cam = self.pose[:3].view(1, 3)
        self.camera.V_cam = self.pose[3:].view(1, 2)
        self.camera.X_cam_history = torch.stack(self.trajectory)[:, :3]
        self.camera.V_cam_history = torch.stack(self.trajectory)[:, 3:]
        return True

    def coverage_mask(self, points=None):
        points = self.points if points is None else points
        if len(points) == 0:
            return np.zeros(len(self.gt_numpy), dtype=bool)
        cloud = points.detach().cpu().numpy() / self.scale
        distances, _ = cKDTree(cloud).query(self.gt_numpy, workers=1)
        return distances <= self.cfg.coverage_threshold_m

    def metrics(self):
        if len(self.points) == 0:
            return dict(coverage=0.0, completion_percent=0.0, completion_cm=float('inf'))
        cloud = self.points.detach().cpu().numpy() / self.scale
        distances, _ = cKDTree(cloud).query(self.gt_numpy, workers=1)
        return dict(coverage=float((distances <= self.cfg.coverage_threshold_m).mean()), completion_percent=float((distances <= 0.05).mean() * 100), completion_cm=float(distances.mean() * 100))

    def obstacle_target(self):
        from .geometry import to_grid
        n = self.cfg.image_size
        obstacle = np.zeros((n, n), dtype=np.float32)
        segments = trimesh.intersections.mesh_plane(self.collision_mesh, [0, 1, 0], [0, float(self.pose[1]), 0])
        center = self.pose.detach().cpu().numpy()[[0, 2]]
        for segment in segments:
            local = -(segment[:, [0, 2]] - center) / self.scale
            length = np.linalg.norm(local[1] - local[0])
            count = max(2, int(length / (self.cfg.extent_m / n) * 2) + 1)
            uv = to_grid(np.linspace(local[0], local[1], count), self.cfg)
            valid = ((uv >= 0) & (uv < n)).all(-1)
            obstacle[uv[valid, 1], uv[valid, 0]] = 1
        return obstacle
