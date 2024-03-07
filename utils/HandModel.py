import json
import os

import numpy as np
import plotly.graph_objects as go
import pytorch_kinematics as pk
import torch
import transforms3d
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from pytorch3d import transforms
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh,
                                                    Sphere)
from tqdm import tqdm, trange
import torch.nn.functional as F
from utils.utils_3d import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_ortho6d, cross_product
from itertools import permutations

class RoboticHand:
    def __init__(self, hand_model, urdf_filename, mesh_path, specs_path=None,
                 batch_size=1, hand_scale=1., pts_density=25000,
                 stl_mesh=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 **kwargs
                 ):
        self.device = device
        self.hand_model = hand_model
        self.batch_size = batch_size
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)

        self.joint_param_names = self.robot.get_joint_parameter_names()
        self.q_len = 9
        
        # prepare geometries for visualization
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.contact_point_dict = json.load(open(os.path.join("data/urdf/", 'contact_%s.json' % hand_model)))
        self.contact_point_basis = {}
        self.contact_normals = {}
        self.surface_points = {}
        
        self.penetration_keypoints_dict = json.load(open(os.path.join("data/urdf/", 'pntr_%s.json' % hand_model)))
        self.penetration_keypoints_dict = { k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in self.penetration_keypoints_dict.items() }
        self.penetration_keypoints_dict = { k: F.pad(v, (0, 1), mode='constant', value=1.0) for k, v in self.penetration_keypoints_dict.items() }
        
        self.hand_keypoints_dict = json.load(open(os.path.join("data/urdf/", 'kpts_%s.json' % hand_model)))
        self.hand_keypoints_dict = { k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in self.hand_keypoints_dict.items() }
        self.hand_keypoints_dict = { k: F.pad(v, (0, 1), mode='constant', value=1.0) for k, v in self.hand_keypoints_dict.items() }
        
        self.n_keypoints = sum([v.shape[0] for v in self.hand_keypoints_dict.values()])
        
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}
        self.num_links = len(visual.links)
        self.num_contacts = len(visual.links)
        self.link_contact_idxs = torch.zeros([len(visual.links)], dtype=torch.long, device=device)
        
        self.contact_to_link_name = []
        self.contact_permutations = [list(p) for p in permutations(np.arange(3))]
                
        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []
        
        self.link_idx_to_pts_idx = {}
        
        self.n_pts = 0
        
        for i_link, link in enumerate(visual.links):
            if len(link.visuals) == 0:
                continue
            
            if type(link.visuals[0].geometry) == Mesh:
                if hand_model == 'shadowhand' or hand_model == 'allegro' or hand_model == 'barrett' or hand_model == 'tonghand' or hand_model == 'tonghand_viz':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                else:
                    filename = link.visuals[0].geometry.filename
                    
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                raise NotImplementedError
            try:
                scale = np.array(link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
                
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])
                
            num_part_pts = int(mesh.area * pts_density)
            self.link_idx_to_pts_idx[link.name] = torch.tensor(np.arange(num_part_pts) + self.n_pts).to(self.device)
            pts = mesh.sample(num_part_pts) * scale
            self.n_pts += num_part_pts

            # Surface Points
            pts = np.matmul(rotation, pts.T).T + translation
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(pts).to(device).float().unsqueeze(0)

            # Visualization Mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)
            
            # Contact Points
            if link.name in self.contact_point_dict:
                cpb = np.array(self.contact_point_dict[link.name])
                
                basis, normals = [], []
                
                for basis_indices in cpb:
                    self.contact_to_link_name.append(link.name)
                    cp_basis = mesh.vertices[basis_indices] * scale
                    cp_basis = np.matmul(rotation, cp_basis.T).T + translation
                    cp_basis = torch.cat([torch.from_numpy(cp_basis).to(device).float(), torch.ones([4, 1]).to(device).float()], dim=-1)
                    basis.append(cp_basis)
                    
                    v1 = cp_basis[1, :3] - cp_basis[0, :3]
                    v2 = cp_basis[2, :3] - cp_basis[0, :3]
                    v1 = v1 / (torch.norm(v1) + 1e-12)
                    v2 = v2 / (torch.norm(v2) + 1e-12)
                    normal = torch.cross(v1, v2).view([1, 3])
                    
                    normals.append(normal)
                
                # N_areas x 4 x 3
                self.contact_point_basis[link.name] = torch.stack(basis, dim=0).unsqueeze(0)
                # N_areas x 3
                self.contact_normals[link.name] = torch.cat(normals, dim=0).unsqueeze(0)
                
            # self.canon_verts.append(torch.tensor(self.mesh_verts[link.name]).to(device).float().unsqueeze(0) * hand_scale)
            # self.canon_faces.append(torch.tensor(mesh.faces).long().to(self.device))
            # self.idx_vert_faces.append(index_vertices_by_faces(self.canon_verts[-1], self.canon_faces[-1]))
            # self.face_normals.append(face_normals(self.idx_vert_faces[-1], unit=True))

        # new 2.1
        self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type == 'revolute' or self.robot_full.joints[i].joint_type == 'continuous':
                self.q_len += 1
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_mid = []
        self.revolute_joints_q_var = []
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i in range(len(self.joint_param_names)):
            for j in range(len(self.revolute_joints)):
                if self.revolute_joints[j].name == self.joint_param_names[i]:
                    joint = self.revolute_joints[j]
            assert joint.name == self.joint_param_names[i]
            self.revolute_joints_q_mid.append((joint.limit.lower + joint.limit.upper) / 2)
            self.revolute_joints_q_var.append(((joint.limit.upper - joint.limit.lower) / 2) ** 2)
            self.revolute_joints_q_lower.append(joint.limit.lower)
            self.revolute_joints_q_upper.append(joint.limit.upper)

        self.revolute_joints_q_mid = torch.Tensor(self.revolute_joints_q_mid).to(device)
        self.revolute_joints_q_lower = torch.Tensor(self.revolute_joints_q_lower).to(device)
        self.revolute_joints_q_upper = torch.Tensor(self.revolute_joints_q_upper).to(device)

        self.current_status = None

        self.canon_pose = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0] + [0] * (self.q_len - 9), device=device, dtype=torch.float32)
        self.scale = hand_scale
        
        self.num_contacts = len(self.contact_to_link_name)
        self.full_cpi = torch.arange(0, self.num_contacts, dtype=torch.long, device=device)
        self.contact_dist_diag_mask = torch.eye(self.num_contacts, dtype=torch.float32, device=device) * 1e12
        self.to_all_contact_areas_cpi = torch.arange(0, self.num_contacts, device=self.device, dtype=torch.long).unsqueeze(0)
        self.full_cpw_zeros = torch.zeros([1, self.num_contacts, 4], dtype=torch.float32, device=device)
        # print(f"[{hand_model}] {self.num_contacts} contact points, {self.n_pts} surface points")

    def random_handcode(self, batch_size, table_top=True):
        transf = torch.normal(0, 1, [batch_size, 9], device=self.device, dtype=torch.float32)
        # joints = torch.rand([batch_size, self.q_len - 9], device=self.device, dtype=torch.float32)
        joints = torch.rand([batch_size, self.q_len - 9], device=self.device, dtype=torch.float32) * 0.5
        joints = joints * (self.revolute_joints_q_upper - self.revolute_joints_q_lower) + self.revolute_joints_q_lower
        q = torch.cat([transf, joints], dim=-1)
        
        q[:, 0:9] = 0.0
        if self.hand_model == "shadowhand":
            q[:, 1] = 0.3
        elif self.hand_model == "tonghand":
            q[:, 2] = 0.3
        
        if table_top:
            R_palm_down = transforms.euler_angles_to_matrix(torch.tensor([torch.pi / 2, 0.0, 0.0]).unsqueeze(0).tile([batch_size, 1]).to(self.device), "XYZ")
            z = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).unsqueeze(0).tile([batch_size, 1])
            v = torch.rand([batch_size, 3], device=self.device, dtype=torch.float32)  
            v = v / torch.norm(v, dim=1).view(-1, 1)  # Normalize
            v[:, -1] = torch.abs(v[:, -1]) + 0.5  # Make sure that the z-component is positive
            axis = torch.cross(z, v)
            axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-12) * torch.acos(torch.clamp(torch.sum(z * v, dim=1), -1, 1)).unsqueeze(-1)
            R_upper_sphere = transforms.axis_angle_to_matrix(axis)
            R = torch.matmul(R_upper_sphere, R_palm_down)
            R6 = compute_ortho6d_from_rotation_matrix(R)
        else:
            R6 = torch.normal(0, 1, [batch_size, 6], dtype=torch.float32, device=self.device)
            R = compute_rotation_matrix_from_ortho6d(R6)
        
        q[:, 0:3] = torch.matmul(R, q[:, 0:3].unsqueeze(-1)).squeeze()
        q[:, 3:9] = R6.clone() + torch.normal(0, 0.1, [batch_size, 6], dtype=torch.float32, device=self.device)
        
        q = q.contiguous().clone()
        q.requires_grad_()
        return q

    def update_kinematics(self, q):
        self.batch_size = q.shape[0]
        self.global_translation = q[:, :3] / self.scale
        self.global_rotation = compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])
        
    def get_contact_points(self, cpi, cpw, q=None):
        cpw = self.softmax(cpw)
        B = cpi.shape[0]
        if q is not None:
            self.update_kinematics(q)
        cpb_trans = []
        for link_name in self.contact_point_basis:
            trans_matrix = self.current_status[link_name].get_matrix().expand([B, 4, 4])
            cp_basis = self.contact_point_basis[link_name].expand([B, self.contact_point_basis[link_name].shape[1], 4, 4])
            N = cp_basis.shape[1]
            
            cp_basis = cp_basis.reshape([B, -1, 4])
            cp_basis = torch.matmul(trans_matrix, cp_basis.transpose(-1, -2)).transpose(-1, -2).reshape([B, N, 4, 4])[..., :3]
            
            cpb_trans.append(cp_basis)
            
        cpb_trans = torch.cat(cpb_trans, 1).contiguous()
        cpb_trans = cpb_trans[torch.arange(0, len(cpb_trans), device=self.device).unsqueeze(1).long(), cpi.long()]
        cpb_trans = (cpb_trans * cpw.unsqueeze(-1)).sum(2)
        cpb_trans = torch.matmul(self.global_rotation, cpb_trans.transpose(-1, -2)).transpose(-1, -2) + self.global_translation.unsqueeze(1)
        
        return cpb_trans * self.scale
    
    def solve_pR(self, target_kpts):
        B, N, _ = target_kpts.shape
        canon_palm_facing_dir = torch.tensor([0., -1., 0.], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1)
        canon_vec_1 = torch.tensor([1.00, 0., 0.], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1)
        canon_pos = torch.tensor([0.0200, 0.0000, 0.0400], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1)

        vec_1 = target_kpts[:, -4] - target_kpts[:, -2]
        vec_2 = target_kpts[:, -1] - target_kpts[:, -2]
        palm_facing_dir = cross_product(vec_1, vec_2)
        palm_facing_dir = palm_facing_dir / (torch.norm(palm_facing_dir, dim=-1, keepdim=True) + 1e-12)

        # 1-stage method requires SVD decomposition, which is slow
        # We use a 2-stage method to separately align the palm-facing direction and the rotation about it.
        rotmat = rotation_of_vectors(canon_palm_facing_dir, palm_facing_dir)

        vec_1 = vec_1 / (torch.norm(vec_1, dim=-1, keepdim=True) + 1e-12)
        palm_facing_vec_1 = torch.matmul(rotmat, canon_vec_1.unsqueeze(-1)).squeeze(-1)
        pf_rotmat = rotation_of_vectors(palm_facing_vec_1, vec_1)
        rotmat = torch.matmul(pf_rotmat, rotmat)

        rot6d = compute_ortho6d_from_rotation_matrix(rotmat)

        pos = target_kpts[:, -4:].mean(dim=1)
        pf_canon_pos = torch.matmul(rotmat, canon_pos.unsqueeze(-1)).squeeze(-1)
        pos = pos - pf_canon_pos
        
        return torch.cat([pos, rot6d], dim=-1)
    
    def fit(self, target_kpts, init=None, n_steps=200, solve_pR=True):
        B = target_kpts.shape[0]
        if init is None:
            init = torch.zeros([B, self.q_len], dtype=torch.float32, device=target_kpts.device)
            init[:, [3, 7]] = 1.0
            
        if solve_pR:
            init[:, :9] = self.solve_pR(target_kpts)

        p = init[:, 0:3].detach().clone().requires_grad_(True)   
        r = init[:, 3:9].detach().clone().requires_grad_(True)   
        j = init[:, 9: ].detach().clone().requires_grad_(True)   
        optimizer = torch.optim.Adam([
            # { "params": p, "lr": 1e-5 },
            # { "params": r, "lr": 1e-5 },
            { "params": j, "lr": 0.1 }
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, 0.01)
        
        for step in trange(n_steps):
            optimizer.zero_grad()
            q = torch.cat([p, r, j], dim=-1)
            cur_kpts = self.get_hand_keypoints(q)
            err = (target_kpts - cur_kpts).norm(dim=-1).mean(dim=-1).sum()
            err.backward()
            optimizer.step()
            scheduler.step()
            if step % 25 == 24:
                tqdm.write(f"Step {step} | Mean L1 (m): {err.item() / B}")
        return q.detach()
    
    def self_penetration(self, q=None):
        points = self.get_penetration_keypoints(q)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).norm(dim=-1)
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        self_pntr_energy = torch.relu(0.015 - dis)
        return self_pntr_energy.sum(dim=[1, 2])

    def get_contact_areas(self, cpi, q=None):
        B, N_c = cpi.shape
        if q is not None:
            self.update_kinematics(q)
            
        ones = torch.ones([B, N_c, 4]).to(cpi.device) * 1e-10
        
        contacts = []
        
        for i in range(4):
            ones_ = ones.clone()
            ones_[:, :, i] = 1e10
            areas, normal = self.get_contact_points_and_normal(cpi, ones_)
            areas = areas + normal * 1e-5
            contacts.append(areas)
            
        return torch.stack(contacts, dim=-2)
    
    def get_contact_points_and_normal(self, cpi, cpw, q=None):
        cpw = self.softmax(cpw)
        if q is not None:
            self.update_kinematics(q)
        B, *_ = cpi.shape
        cpb_trans, cpn_trans = [], []
        for link_name in self.contact_point_basis:
            trans_matrix = self.current_status[link_name].get_matrix().expand([self.batch_size, 4, 4])
            cp_basis = self.contact_point_basis[link_name]
            cp_normal = self.contact_normals[link_name]
            N = cp_basis.shape[1]
            
            cp_basis = cp_basis.expand([self.batch_size, cp_basis.shape[1], 4, 4]).reshape([self.batch_size, -1, 4])
            cp_normal = cp_normal.expand([self.batch_size, cp_normal.shape[1], 3])
            
            cp_basis = torch.matmul(trans_matrix, cp_basis.transpose(-1, -2)).transpose(-1, -2).reshape([self.batch_size, N, 4, 4])[..., :3]
            cp_normal = torch.matmul(trans_matrix[..., :3, :3], cp_normal.transpose(-1, -2)).transpose(-1, -2).reshape([self.batch_size, N, 3])
            
            cpb_trans.append(cp_basis)
            cpn_trans.append(cp_normal)
            
        cpb_trans = torch.cat(cpb_trans, 1).contiguous()
        cpb_trans = cpb_trans[torch.arange(0, len(cpb_trans), device=self.device).unsqueeze(1).long(), cpi.long()]
        cpb_trans = (cpb_trans * cpw.unsqueeze(-1)).sum(2)
        cpb_trans = torch.matmul(self.global_rotation, cpb_trans.transpose(-1, -2)).transpose(-1, -2) + self.global_translation.unsqueeze(1)
        
        cpn_trans = torch.cat(cpn_trans, 1).contiguous()
        cpn_trans = cpn_trans[torch.arange(0, len(cpn_trans), device=self.device).unsqueeze(1).long(), cpi.long()]
        cpn_trans = torch.matmul(self.global_rotation, cpn_trans.transpose(1, 2)).transpose(1, 2)
        
        return cpb_trans * self.scale, cpn_trans
    
    def prior(self, q):
        range_energy = torch.relu(q[:, 9:] - self.revolute_joints_q_upper) + torch.relu(self.revolute_joints_q_lower - q[:, 9:])
        return range_energy.sum(-1)

    def get_vertices(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            trans_matrix = self.current_status[link_name].get_matrix().expand([self.batch_size, 4, 4])
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].repeat(self.batch_size, 1, 1).transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points[0] = surface_points[0].expand([self.batch_size, surface_points[0].shape[1], 3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale
    
    def get_penetration_keypoints(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        kpts = []

        for link_name, canon_kpts in self.penetration_keypoints_dict.items():
            trans_matrix = self.current_status[link_name].get_matrix().expand([self.batch_size, 4, 4])
            kpts.append(torch.matmul(trans_matrix, canon_kpts.unsqueeze(0).expand([self.batch_size, canon_kpts.shape[0], canon_kpts.shape[1]]).transpose(-1, -2)).transpose(-1, -2)[..., :3])
        kpts = torch.cat(kpts, 1)
        kpts = torch.matmul(self.global_rotation, kpts.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return kpts * self.scale
    
    def get_hand_keypoints(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        kpts = []

        for link_name, canon_kpts in self.hand_keypoints_dict.items():
            trans_matrix = self.current_status[link_name].get_matrix().expand([self.batch_size, 4, 4])
            kpts.append(torch.matmul(trans_matrix, canon_kpts.unsqueeze(0).expand([self.batch_size, canon_kpts.shape[0], canon_kpts.shape[1]]).transpose(-1, -2)).transpose(-1, -2)[..., :3])
        kpts = torch.cat(kpts, 1)
        kpts = torch.matmul(self.global_rotation, kpts.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return kpts * self.scale
    
    def get_meshes_from_q(self, q=None, i=0, concat=True):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(), transformed_v.T).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        if concat:
            data = tm.util.concatenate(data)
        return data

    def get_plotly_data(self, q=None, i=0, concat=True, color='lightpink', opacity=1.0, name='tonghand'):
        mesh = self.get_meshes_from_q(q)
        if concat:
            mesh = tm.util.concatenate(mesh)
            return go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color=color, opacity=opacity, name=name
            )
        else:
            return [
                go.Mesh3d(
                    x=m.vertices[:, 0],
                    y=m.vertices[:, 1],
                    z=m.vertices[:, 2],
                    i=m.faces[:, 0],
                    j=m.faces[:, 1],
                    k=m.faces[:, 2],
                    color=color, opacity=opacity, name=f"{name}_{i_m}"
                ) for i_m, m in enumerate(mesh)
            ]

robotic_hand_files = {
    "shadowhand":
    {
        "urdf_filepath": 'data/urdf/shadow_hand_description/shadowhand.urdf',
        "mesh_filepath": 'data/urdf/shadow_hand_description/meshes',
    }
}
    
def get_hand_model(hand_model, batch_size, device='cuda', **kwargs) -> RoboticHand:
    with torch.no_grad():
        filepaths = robotic_hand_files[hand_model]
        hand_model = RoboticHand(hand_model, filepaths['urdf_filepath'], filepaths['mesh_filepath'], specs_path=filepaths['specs'] if 'specs' in filepaths else None, batch_size=batch_size, device=device, **kwargs)
        
    return hand_model