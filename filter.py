from collections import defaultdict
import numpy as np
import argparse
from copy import deepcopy
import enum
import json
from math import radians
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pytorch3d import transforms
import torch.nn.functional as F
import torch
import trimesh as tm
from tqdm import tqdm, trange
from utils.ObjectModels import MeshModel
from utils.Snapshot import load_from_snapshot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils.utils import fps_indices
from utils.utils_3d import do_rotation, do_translation, compute_rotation_matrix_from_ortho6d, tensor_linspace, undo_rotation

from utils.visualize_plotly import plot_grasps, plot_mesh, plot_point_cloud, plot_rect

sns.set(rc={"figure.figsize": (16, 9)})
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def accept(fc_loss, pen, surf_dist, hand_pri, v=False):
    fc_thres  = 0.25
    sd_thres  = 0.005
    pen_thres = 0.005
    hp_thres  = 0.1
    
    surf_dist = surf_dist.mean(-1).max(-1)[0]
    fc_loss = fc_loss.max(dim=-1)[0]
    pen = pen.mean(dim=-1)
    
    fc_acc  = fc_loss   < fc_thres
    sd_acc  = surf_dist < sd_thres
    pen_acc = pen       < pen_thres
    hp_acc  = hand_pri  < hp_thres

    accepted = fc_acc * pen_acc * sd_acc * hp_acc
    
    if v:
        return accepted.float(), fc_acc.float(), pen_acc.float(), sd_acc.float(), hp_acc.float()
    else:
        tqdm.write(f"Total: { accepted.float().sum() } FC: { fc_acc.float().sum() }, PN: { pen_acc.float().sum() }, SD: { sd_acc.float().sum() }, HP: { hp_acc.float().sum() }")

        plt.subplot(3, 1, 1)
        plt.title("FC")
        plt.hist(fc_loss.cpu().numpy(), np.linspace(0, 1.0, 50))
        plt.subplot(3, 1, 2)
        plt.title("distance")
        plt.hist(surf_dist.cpu().numpy(), np.linspace(0, 0.04, 50))
        plt.subplot(3, 1, 3)
        plt.title("penetration")
        plt.hist(pen.cpu().numpy(), np.linspace(0, 0.04, 50))
        return accepted.float()

if __name__ == "__main__":
    ds_tag = "GraspEm"
        
    paths = [
        "synthesis/shadowhand/2024-03/07/23-50-59_duck+cylinder-seed_42-demo",
    ]

    paths = sorted(paths)

    export_path = os.path.join("synthesis", "export")
    os.makedirs(export_path, exist_ok=True)
    grasps = []
    filtered_grasps = []

    uuids_list = []
    objects_list = []
    obj_transl_list = []
    obj_orient_list = []
    obj_params_list = []
    obj_scales_list = []
    joint_mask_list = []
    q_list, cpi_list, cpw_list = [], [], []

    hc_list = []
    fc_list = []
    o_c_list, o_r_list = [], []
    energy_list = []

    parent_ckpt_count = {}

    object_indices = defaultdict(list)
    total_acc = 0
        
    for path in tqdm(paths):
        with torch.no_grad():
            ckpt_path = os.path.join(path, "ckpt.pt")
            if not os.path.exists(ckpt_path):
                continue
            
            uuids, args, q, cpi, cpw, physics_guide, hand_model, quantitative = load_from_snapshot(path=ckpt_path, batch_rec=False)
            n_objects = len(args.object_models)
            
            fc_error = quantitative['fc_loss']
            pen = quantitative['pen']
            surf_dist = quantitative['surf_dist']
            hand_pri = quantitative['hand_pri']
            norm_ali = quantitative['norm_ali']
            
            plt.figure()
            acc = accept(fc_error, pen, surf_dist, hand_pri)
            acc_idx = torch.where(acc)[0].long()
            plt.tight_layout()
            plt.savefig(os.path.join(path, "stat.png"))
                    
            tqdm.write(f"Accepted { acc_idx.shape[0] } out of { acc.shape[0] } grasps from checkpoint { path }")
            if acc_idx.shape[0] == 0:
                continue
            
            obj_names = [ obj.object_model for obj in physics_guide.object_models ]
            
            total_acc += acc_idx.shape[0]
            
            n_plot = min(acc_idx.shape[0], 16)
            plot_indices = acc_idx[:n_plot]
            
            viz_dir = os.path.join(path, "plot")
            stl_dir = os.path.join(path, "stl")
            os.makedirs(viz_dir, exist_ok=True)
            os.makedirs(stl_dir, exist_ok=True)
            
            hand_model.update_kinematics(q)
            contacts = [ hand_model.get_contact_areas(cpi[:, j]).cpu().numpy() for j in range(n_objects) ]
            contact_pts = [ hand_model.get_contact_points(cpi[:, j], cpw[:, j]).cpu().numpy() for j in range(n_objects) ]
            pntr_kpts = hand_model.get_penetration_keypoints().cpu().numpy()
            
            for i in range(n_plot):
                data_i = plot_indices[i]
                o_meshes = [ o.get_obj_mesh(data_i) for j, o in enumerate(physics_guide.object_models) ]
                h_mesh = hand_model.get_meshes_from_q(None, data_i, True)
                go.Figure([ plot_mesh(m, name=f"object-{j}") for j, m in enumerate(o_meshes) ]
                # + [ plot_point_cloud(pntr_kpts[data_i], name=f"pntr-kpts", color='blue') ]
                + [ plot_point_cloud(contact_pts[j][data_i], name=f"contact-{j}") for j in range(n_objects) ]
                + [ plot_rect(contacts[j][data_i, k], name=f"contact-{j}_{k}", color='red') for j in range(n_objects) for k in range(args.n_contact) ]
                + [ plot_mesh(tm.load("data/table.stl"), color='green'), plot_mesh(h_mesh, 'lightpink', opacity=1.0) ]
                ).write_html(os.path.join(viz_dir, f"{data_i}_{uuids[data_i]}.html"))
                
                h_mesh.export(os.path.join(stl_dir, f"{uuids[data_i]}_h.stl"))
                for i_o, o in enumerate(o_meshes):
                    o.export(os.path.join(stl_dir, f"{uuids[data_i]}_o-{i_o}.stl"))
                    
            obj_poses = [obj.get_poses() for obj in physics_guide.object_models]
            uuids_list += [ uuids[i] for i in acc_idx ]

            obj_transl_list.append([pose[0][acc_idx] for pose in obj_poses])
            obj_orient_list.append([pose[1][acc_idx] for pose in obj_poses])
            obj_scales_list.append([obj.scale for obj in physics_guide.object_models])
            objects_list += [[obj.object_model for obj in physics_guide.object_models]] * len(acc_idx)
            q_list.append(q[acc_idx])
            cpi_list.append(cpi[acc_idx])
            cpw_list.append(cpw[acc_idx])
            
            for l in locals():
                del l
            torch.cuda.empty_cache()

    transls = torch.cat([torch.stack(transl, dim=1) for transl in obj_transl_list], dim=0)
    orients = torch.cat([torch.stack(orient, dim=1) for orient in obj_orient_list], dim=0)
    scales = torch.cat([torch.stack(scale, dim=1) for scale in obj_scales_list], dim=0)

    cpi = torch.cat(cpi_list, dim=0)
    cpw = torch.cat(cpw_list, dim=0)
    q = torch.cat(q_list)
    
    # Palm alignment towards the Y+ direction
    with torch.no_grad():
        hand_model.update_kinematics(q)
        kpts = hand_model.get_hand_keypoints()
        p = kpts[:, -4:].mean(dim=1)
        p[..., -1] = 0
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-12)
        desired_dir = torch.tensor([0.0, 1.0, 0.0], device=p.device).expand(p.shape[0], -1)
        rot_angle = torch.arccos(p[:, 0])
        
        hand_p = q[:, :3].clone()
        hand_r = transforms.matrix_to_quaternion(compute_rotation_matrix_from_ortho6d(q[:, 3:9].clone()))
        hand_r = torch.cat([hand_r[..., 1:], hand_r[..., :1]], dim=-1)
        hand_j = q[:, 9:].clone()

        torch.save({
            "uuids": uuids_list,
            "objects": objects_list,
            "object_scale": scales,
            "q": q,
            "cpi": cpi,
            "cpw": cpw,
            "obj_p": transls,
            "obj_r": orients,
        }, os.path.join(export_path, f"GraspEm.2.pt"))
    