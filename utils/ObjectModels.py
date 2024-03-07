'''
LastEditors: Aiden Li (i@aidenli.net)
Description: Object models and thier computations
Date: 2022-01-24 01:45:04
LastEditTime: 2022-07-15 00:27:05
Author: Aiden Li
'''
from copy import deepcopy
import json
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh as tm
from torch import normal
from torch.nn.functional import normalize
from tqdm import trange
from utils.ODField import ODField

from utils.utils import *
from utils.utils_3d import *
from utils.visualize_plotly import *


class ObjectModel(metaclass=ABCMeta):
    def __init__(self):
        pass
        
    @abstractmethod
    def _sample_pts(self, num_pts=512):
        """
        Sample points on the object.
        
        Used for penetration detection, distance calculation.
        Used as contact points.
        """
        pass
        
    @abstractmethod
    def po_penetration(self, points, self_center=None, self_rotation=None):
        """
        Penetration from a point cloud to the objects.
        """
        pass
    
    @abstractmethod
    def mo_penetration(self, points, self_center=None, self_rotation=None):
        """
        Penetration from a mesh to the objects.
        """
        pass
    
    @abstractmethod
    def distance(self, points):
        """
        Calculate distances from query points x to the object
        """
        pass
    
    @abstractmethod
    def distance_gradient(self, points):
        """
        Calculate the gradient of distance, indicating the best direction of approching the object.
        """
        pass
    
    @abstractmethod
    def get_plot(self, idx, color='lightblue', opacity=0.9):
        """
        Get mesh of the object.
        """
        pass
    
    def update_pose(self, transl, orient):
        pass
    
class SphereModel(ObjectModel):
    def __init__(self, object_model, radius, batch_size, num_pts=256, device='cuda'):
        super().__init__()
        self.object_model = object_model
        self.radius = radius
        self.scale = radius
        self.batch_size = batch_size
        self.num_pts = num_pts
        
        self.device = device
        
        self.to_local = lambda pts: undo_translation(pts, self.transl)
        self.from_local = lambda pts: do_translation(pts, self.transl)

        with torch.no_grad():
            self.transl = torch.zeros([self.radius.shape[0], 3], device=device).float()
            self.orient = torch.eye(3, device=device).unsqueeze(0).tile((self.radius.shape[0], 1, 1)).float()
            self.surface_points = self._sample_pts()

    def get_surface_points(self):
        return self.from_local(self.surface_points)
            
    def _sample_pts(self):
        pts = tm.primitives.Sphere(radius=1.0).sample(self.num_pts)
        pts = torch.tensor(pts, dtype=torch.float32, device=self.device).unsqueeze(0).tile(self.radius.shape[0], 1, 1)
        pts = pts * self.radius.unsqueeze(-1).unsqueeze(-1).expand(pts.shape)
        return pts

    def distance(self, points):
        points = self.to_local(points)
        return torch.norm(points, dim=-1) - self.radius.unsqueeze(-1)
        
    def po_penetration(self, points):
        dist = self.distance(points)
        return torch.relu(-dist)
    
    def mo_penetration(self, mesh):
        points = mesh_to_pts(mesh).to(self.device)
        return self.po_penetration(points)
    
    def distance_gradient(self, points):
        points = self.to_local(points)
        return points.norm(dim=-1) - self.radius.unsqueeze(-1), points / points.norm(dim=-1, keepdim=True)
    
    def update_pose(self, transl, orient):
        self.transl = transl
        # self.orient = orient

    def get_obj_mesh(self, index):
        return tm.primitives.Sphere(radius=self.radius[index].item(), center=self.transl[index].cpu().numpy())

    def get_plot(self, idx, color='lightblue', opacity=1.0):
        spheres = tm.primitives.Sphere(radius=self.radius[idx], center=self.transl[idx].detach().cpu().numpy())
        return go.Mesh3d(
            x=spheres.vertices[:, 0],
            y=spheres.vertices[:, 1],
            z=spheres.vertices[:, 2],
            i=spheres.faces[:, 0],
            j=spheres.faces[:, 1],
            k=spheres.faces[:, 2],
            color=color, opacity=opacity)
        
    def batch_filter(self, idx):
        self.batch_size = idx.shape[0]
        self.radius = self.radius[idx]
        self.transl = self.transl[idx]
        self.orient = self.orient[idx]
        # self.sampled_pts = self.sampled_pts[idx]
        
    def batch_tile(self, factor):
        self.batch_size *= factor
        self.radius = self.radius.tile((factor))
        self.transl = self.transl.tile((factor, 1))
        self.orient = self.orient.tile((factor, 1, 1))
        # self.sampled_pts = self.sampled_pts.tile((factor, 1, 1))
        
    def get_poses(self):
        return [self.transl, self.orient]
    
    def get_params(self):
        return self.radius
    
    def set_params(self, params):
        self.radius = params.to(self.device)


class ODFieldModel():
    def __init__(self, object_model, batch_size, num_pts=256, scale=1.0, device='cuda', signed=True):
        self.device = device
        self.object_model = object_model
        self.num_pts = num_pts
        self.batch_size = batch_size
        # if dataset == "the" :
        self.sdfield = ODField().to(device)
        # else:        
        #     self.sdfield = ODField().to(device)
        self.sdfield.eval()
        self.signed = signed
        self.scale = scale
        basedir="data/objects"
        self.obj_names = json.load(open(os.path.join(basedir, "names.json"), 'r'))
        meshpath = self.obj_names[object_model]
        meshpath = os.path.join(basedir, meshpath)
        self.data_dir = os.path.dirname(meshpath)
        self.mesh = tm.load_mesh(meshpath, force='mesh')
        
        self.sdfield.load_state_dict(torch.load(os.path.join(self.data_dir, f"{'sdfield' if signed else 'udfield'}.pt")))
        self.stable_rotations = torch.tensor(json.load(open(os.path.join(basedir, 'rotations_filtered.json'), 'r'))[object_model], dtype=torch.float32, device=self.device)
        self.load_mesh(self.mesh)
        
        stable_rot_pts = torch.tensor(self.mesh.sample(1024)).float().to(self.device).unsqueeze(0).tile([self.stable_rotations.shape[0], 1, 1])
        stable_rot_pts = torch.matmul(self.stable_rotations, stable_rot_pts.transpose(-1, -2)).transpose(-1, -2)
        self.stable_zs = - stable_rot_pts[..., 2].min(dim=1)[0]
            
        self.to_local = lambda pts: undo_rotation((undo_translation(pts, self.transl)), self.orient)
        self.from_local = lambda pts: do_translation(do_rotation(pts, self.orient), self.transl)
        
        with torch.no_grad():
            self.transl = torch.zeros([self.batch_size, 3], device=device).float()
            self.orient = torch.eye(3, device=device).unsqueeze(0).tile((self.batch_size, 1, 1)).float()
            self.surface_points = self._sample_pts()
            
    def load_mesh(self, mesh):
        pts_surf = torch.tensor(mesh.sample(512)).float().to(self.device).unsqueeze(0)
        self.sampled_points = pts_surf
        self.canon_object_verts = torch.Tensor(mesh.vertices).to(self.device).unsqueeze(0)
        self.object_faces = torch.Tensor(mesh.faces).long().to(self.device)

    def index_mesh_names(self, names):
        return torch.tensor([ list(self.obj_names.keys()).index(name) for name in names ], dtype=torch.long).to(self.device)
            
    def _sample_pts(self):
        pts = torch.tensor(self.mesh.sample(self.num_pts), dtype=torch.float32, device=self.device).unsqueeze(0).tile((self.batch_size, 1, 1))
        return pts * self.scale.unsqueeze(-1).unsqueeze(-1)
    
    def get_surface_points(self):
        return self.from_local(self.surface_points)
    
    def distance(self, x):
        x = self.to_local(x) / self.scale.unsqueeze(-1).unsqueeze(-1)
        return self.sdfield(x)[..., 0] * self.scale.unsqueeze(-1)

    def gradient(self, x):
        x = self.to_local(x) / self.scale.unsqueeze(-1).unsqueeze(-1)
        gradient = self.sdfield(x)[..., 1:]
        return gradient / (torch.norm(gradient, dim=-1, keepdim=True) + 1e-12)

    def distance_gradient(self, x):
        x = self.to_local(x) / self.scale.unsqueeze(-1).unsqueeze(-1)
        field = self.sdfield(x)
        distance = field[..., 0] * self.scale.unsqueeze(-1)
        gradient = field[..., 1:]
        return distance, gradient / (torch.norm(gradient, dim=-1, keepdim=True) + 1e-12)

    def po_penetration(self, x):
        return torch.relu(- self.distance(x))
    
    def get_obj_mesh(self, idx):
        mesh = deepcopy(self.mesh)
        verts = torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32).unsqueeze(0) * self.scale[idx]
        verts = do_translation(do_rotation(verts, self.orient[idx:idx+1]), self.transl[idx:idx+1])
        mesh.vertices = verts[0].cpu().numpy()
        return mesh
    
    def update_pose(self, transl, orient):
        self.transl = transl
        self.orient = orient
    
    def batch_filter(self, idx):
        self.batch_size = idx.shape[0]
        self.scale = self.scale[idx]
        self.transl = self.transl[idx]
        self.orient = self.orient[idx]
        self.surface_points = self.surface_points[idx]
        
    def batch_tile(self, factor):
        self.batch_size *= factor
        self.scale = self.scale.tile((factor))
        self.transl = self.transl.tile((factor, 1))
        self.orient = self.orient.tile((factor, 1, 1))
        self.surface_points = self.surface_points.tile((factor, 1, 1))

    def get_plot(self, idx, color='lightblue', opacity=1.0):
        mesh = self.get_obj_mesh(idx)
        return go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color=color, opacity=opacity)
        
    def get_poses(self):
        return [self.transl, self.orient]
    
    def get_params(self):
        return self.object_model
    
    def set_params(self, param):
        self.object_model = param

def get_object_model(object_model, batch_size, scale=1.0, device='cuda'):
    if object_model == "sphere":
        return SphereModel(object_model, scale, batch_size, device=device)
    else:
        if not isinstance(scale, torch.Tensor) and scale == 0.0:
            scale = torch.linspace(0.02, 0.035, batch_size, dtype=torch.float32, device=device)
        elif not isinstance(scale, torch.Tensor):
            scale = torch.ones([batch_size], dtype=torch.float32, device=device) * scale
        return ODFieldModel(object_model, batch_size=batch_size, scale=scale, device=device)
   
class MeshModel():
    def __init__(self, batch_size, object_model=None, num_pts=1024, obj_scale=1.0, device='cuda'):
        from kaolin.metrics.trianglemesh import point_to_mesh_distance
        from kaolin.ops.mesh import check_sign, index_vertices_by_faces, face_normals
        
        self.name = object_model
        self.device = device

        self.batch_size = batch_size
        self.num_pts = num_pts
        self.obj_scale = obj_scale

        basedir="data/objects"
        self.obj_names = json.load(open(os.path.join(basedir, "names.json"), 'r'))
        meshpath = self.obj_names[object_model]
        meshpath = os.path.join(basedir, meshpath)
        self.data_dir = os.path.dirname(meshpath)
        mesh = tm.load_mesh(meshpath, force='mesh')
        self.mesh = mesh
        self.load_mesh(mesh)

    def load_mesh(self, mesh):
        """
        Note that for parallel computation in kaolin, the mesh is loaded as 1-batch.
        3D computations are carried out in the object-centered frame.
        """
        self.mesh = mesh
        self.canon_object_verts = torch.Tensor(mesh.vertices).to(self.device).unsqueeze(0)
        self.object_faces = torch.Tensor(mesh.faces).long().to(self.device)
        self.object_face_verts = index_vertices_by_faces(self.canon_object_verts, self.object_faces)
        self.object_verts = self.canon_object_verts.clone()
        self.face_normals = face_normals(self.object_face_verts, unit=True)

    def distance(self, points, signed=False):
        B, N, _ = points.shape
        points = points.reshape([1, -1, 3])
        dis, _, _ = point_to_mesh_distance(points, self.object_face_verts)
        if signed:
            signs = check_sign(self.object_verts, self.object_faces, points)
            dis = torch.where(signs, -dis, dis)
        return torch.sqrt(dis).reshape([B, N])

    def gradient(self, points, distance):
        grad = torch.autograd.grad([distance.sum()], [points], create_graph=True, allow_unused=True)[0]
        return grad / grad.norm(dim=-1, keepdim=True)

    def distance_gradient(self, points, signed=False):
        B, N, _ = points.shape
        points = points.reshape([1, -1, 3])
        dis, face_idx, _ = point_to_mesh_distance(points, self.object_face_verts)
        dis = torch.sqrt(dis)

        if signed:
            signs = check_sign(self.object_verts, self.object_faces, points)
            dis = torch.where(signs, -dis, dis)

        gradient = torch.autograd.grad([dis.sum()], [points], allow_unused=False, create_graph=False, retain_graph=False)[0]
        gradient = gradient / (torch.norm(gradient, dim=-1, keepdim=True) + 1e-12)

        return dis, gradient.reshape([B, N, 3])

    def _sample_pts(self, num_pts=512):
        return torch.Tensor(self.mesh.sample(num_pts)).to(self.device).unsqueeze(0).tile((self.batch_size, 1, 1))

    def get_obj_mesh(self):
        return deepcopy(self.mesh)

    def get_plot(self, color='lightblue', opacity=1.0):
        vertices = self.object_verts.clone().detach()[0].cpu()
        faces = self.mesh.faces
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color, opacity=opacity)
