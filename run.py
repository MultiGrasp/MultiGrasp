
'''
Author: Aiden Li
Date: 2022-05-24 21:22:43
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-07-15 01:11:09
Description: Grasp Synthesis with MCMC with contact point weights
'''
import argparse
import json
import os
from pytorch3d import transforms
import random
from datetime import datetime
from uuid import uuid4
import torch.nn.functional as F
import numpy as np
import torch
import trimesh as tm
from plotly import graph_objects as go
from tqdm import tqdm, trange
from hand_consts import get_contact_pool, contact_groups

from utils.HandModel import get_hand_model
from utils.ObjectModels import get_object_model
from utils.PhysicsGuide import PhysicsGuide
from utils.utils import *
from utils.visualize_plotly import plot_mesh, plot_point_cloud, plot_rect
from tensorboardX import SummaryWriter

from loguru import logger

from torch.optim.adam import Adam

def parse_args():
    parser = argparse.ArgumentParser()
    # Computation
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--max_physics', default=6500, type=int)
    parser.add_argument('--max_refine', default=1500, type=int)

    # Task settings - Hand
    parser.add_argument('--hand_model', default='shadowhand', type=str)
    parser.add_argument('--n_contact', default=3, type=int)

    # Task settings - Object
    parser.add_argument('--object_models', nargs='+', default=['knob', 'cylinder'], type=str)
    parser.add_argument('--num_obj_pts', default=256, type=int)
    
    # MCMC params
    parser.add_argument('--starting_temperature', default=8., type=float)
    parser.add_argument('--contact_switch', default=0.25, type=float)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--stepsize_period', default=100, type=int)
    parser.add_argument('--annealing_period', default=50, type=int)
    parser.add_argument('--contact_group', type=int, default=0)
    parser.add_argument('--langevin_probability', default=0.85, type=float)
    parser.add_argument('--noise_size', default=0.01, type=float)
    
    # Metric Weights
    parser.add_argument('--fc_error_weight', default=1.0, type=float, help="Weight for force-closure error energy")
    parser.add_argument('--hprior_weight', default=10.0, type=float, help="Weight for hand prior energy")
    parser.add_argument('--pen_weight', default=10.0, type=float, help="Weight for hand-object penetraiton energy")
    parser.add_argument('--sf_dist_weight', default=10.0, type=float, help="Weight for hand-object surface distance energy")
    parser.add_argument('--hc_pen', action='store_true', help="Enable hand self-penetration energy")
    parser.add_argument('--viz', action='store_true', help="Visualize periodically")
    parser.add_argument('--log', action='store_true', help="Log information periodically")
    
    parser.add_argument('--levitate', action='store_true', help="Grasping levitate objects, rather than on the tabletop")
    
    # Debugging
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='synthesis', type=str)
    parser.add_argument('--tag', default='debug', type=str)
    
    return parser.parse_args()

    
def initialize(args):
    # Computation device
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Time tag and export directories
    time_tag = datetime.now().strftime('%Y-%m/%d/%H-%M-%S')
    base_dir = f"{ str(args.output_dir) }/{ str(args.hand_model) }/{ time_tag }_{ '+'.join(args.object_models) }-seed_{args.seed}-{args.tag}"
    
    logger.add(os.path.join(base_dir, "log.txt"), rotation="10 MB", format="{time} {level} {message}")
    logger.info(f"Logging to { os.path.join(base_dir, 'log.txt') }")
    
    os.makedirs(base_dir, exist_ok=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    writer = SummaryWriter(logdir=base_dir)
    
    json.dump(vars(args), open(os.path.join(base_dir, "args.json"), 'w'), default=lambda o: str(o))
    
    return { 
        "writer": writer,
        "base_dir": base_dir,
        "time_tag": time_tag,
    }
    
def synthesis(args, export_configs):
    writer = export_configs['writer']
    n_objects = len(args.object_models)
    
    transl_decay = 1.0
    export_dir = export_configs['base_dir']
    uuids = [ str(uuid4()) for _ in range(args.batch_size) ]

    contact_group = contact_groups[args.contact_group]
    contact_pools = [ torch.tensor(get_contact_pool(g), dtype=torch.long, device=args.device) for g in contact_group ][:n_objects]
    
    logger.info("> Loading models...")
    hand_model = get_hand_model(args.hand_model, args.batch_size, device=args.device)
    physics_guide = PhysicsGuide(hand_model, args)
    
    for obj_name in args.object_models:
        object_model = get_object_model(obj_name, args.batch_size, scale=1.0, device=args.device)
        physics_guide.append_object(object_model)
        
    logger.info(f"  + Hand: { args.hand_model } ( {hand_model.num_links} links, {hand_model.num_contacts} contacts, {hand_model.n_pts} points )")
    logger.info(f"  + Contact groups: { contact_group }")
    logger.info(f"  + Objects: ({ ', '.join(args.object_models) })")
    logger.info(f"  + Export directory: { export_dir }")
    
    def sum_energy(new_fc_error, new_sf_dist, new_pntr, new_hprior, low_fc_w=False):
        new_energy = 0
        new_energy = new_energy + new_fc_error * args.fc_error_weight
        new_energy = new_energy + new_sf_dist * args.sf_dist_weight
        new_energy = new_energy + new_pntr * args.pen_weight
        new_energy = new_energy + new_hprior * args.hprior_weight
                    
        return new_energy
    
    logger.info("> Initializing hand...")
    q = hand_model.random_handcode(args.batch_size, table_top=True)
    
    cpi = torch.randint(0, len(hand_model.contact_point_dict), [args.batch_size, n_objects, args.n_contact], device=args.device, dtype=torch.long)
    cpw = torch.normal(0, 1, [args.batch_size, n_objects, args.n_contact, 4], requires_grad=True, device=args.device).float()
    
    cpi = []
    for i_contact_set, contact_pool in enumerate(contact_pools):
        _contact_pool = contact_pool.unsqueeze(0).tile([args.batch_size, 1])
        cpi_i = torch.randint(0, len(contact_pool), [args.batch_size, args.n_contact], device=args.device, dtype=torch.long)
        contacts = torch.gather(_contact_pool, 1, cpi_i)
        cpi.append(contacts)
    cpi = torch.stack(cpi, dim=1)
    
    logger.info("> Initializing objects...")
    object_ps = []
    object_rs = []
    n_object_poses = 0
    
    # Sample table-top object placement until the batch_size is fulfilled
    while True:
        if len(args.object_models) > 3:
            proposals_xyz = torch.rand([args.batch_size, n_objects, 3], device=args.device) * 0.15 - 0.075
        else:
            proposals_xyz = torch.rand([args.batch_size, n_objects, 3], device=args.device) * 0.075 - 0.0375
        
        for i_object, object_model in enumerate(physics_guide.object_models):
            if object_model.object_model == "sphere":
                proposals_xyz[:, i_object, 2] = object_model.radius
                proposals_rot = torch.eye(3, device=args.device).unsqueeze(0).repeat([args.batch_size, 1, 1])
            else:
                proposals_rot_i = torch.randint(0, len(object_model.stable_rotations), [args.batch_size], device=args.device, dtype=torch.long)
                proposals_rot = object_model.stable_rotations[proposals_rot_i]
                random_z_rot = torch.rand([args.batch_size], dtype=torch.float32, device=args.device) * 2 * torch.pi
                random_z_rot_axis = F.pad(random_z_rot.unsqueeze(-1), (2, 0), 'constant', 0)
                random_z_rot_mat = transforms.axis_angle_to_matrix(random_z_rot_axis)
                proposals_rot = torch.matmul(random_z_rot_mat, proposals_rot)
                proposals_xyz[:, i_object, 2] = object_model.stable_zs[proposals_rot_i]
            object_model.update_pose(proposals_xyz[:, i_object], proposals_rot)
        
        oo_pen = physics_guide.oo_penetration()
        selected = torch.where(oo_pen < 0.0001)[0]
        n_append = min(len(selected), args.batch_size - n_object_poses)
        n_object_poses += n_append
        selected = selected[:n_append]
        object_ps.append(proposals_xyz[selected].clone())
        object_rs.append(torch.stack([ o.orient for o in physics_guide.object_models], dim=1)[selected].clone())
        if n_object_poses == args.batch_size:
            object_ps = torch.cat(object_ps, dim=0).contiguous()
            object_rs = torch.cat(object_rs, dim=0).contiguous()
            for i_object, object_model in enumerate(physics_guide.object_models):
                object_model.update_pose(object_ps[:, i_object], object_rs[:, i_object])
            break
        
    torch.cuda.empty_cache()
    
    logger.info("> Starting optimization...")
    
    fc_error, sf_dist, pntr, hprior, norm_ali = physics_guide.compute_energy(cpi, cpw, q)
    energy_grad = sum_energy(fc_error, sf_dist, pntr, hprior, low_fc_w=True)
    energy = sum_energy(fc_error, sf_dist, pntr, hprior)

    grad_q, grad_w = torch.autograd.grad(energy_grad.sum(), [q, cpw], allow_unused=True)
    grad_q[:, :9] = grad_q[:, :9] * transl_decay
    
    # Steps with contact adjustment
    for step in tqdm(range(args.max_physics)):
        step_size = physics_guide.get_stepsize(step)
        temperature = physics_guide.get_temperature(step)
        new_q = q.clone()
        new_cpi = cpi.clone()
        
        q_grad_weight = max(args.max_physics * 0.8 - step, 0) / args.max_physics * 75 + 25
        ones = torch.arange(0, args.batch_size, device=args.device).long()
        # Updating handcode
        grad_q[:, 9:] = grad_q[:, 9:] / (physics_guide.grad_ema_q.average.unsqueeze(0) + 1e-12)
        noise = torch.normal(mean=0, std=args.noise_size, size=new_q.shape, device='cuda', dtype=torch.float) * step_size # * disabled_joint_mask
        new_q = new_q + (noise - 0.5 * grad_q * step_size * step_size) * physics_guide.joint_mask
        
        # Updating contact point indices (cpi)
        switch_contact = np.random.rand(1) < args.contact_switch
        if switch_contact:
            for i_contact, contact_pool in enumerate(contact_pools):
                update_indices = torch.randint(0, args.n_contact, size=[args.batch_size], device=args.device)
                update_to = torch.randint(0, contact_pool.shape[0], size=[args.batch_size], device=args.device)
                update_to = contact_pool[update_to]
                new_cpi[ones, i_contact, update_indices.long()] = update_to

        cpw = cpw - 0.5 * grad_w * step_size * step_size
        new_fc_error, new_sf_dist, new_pntr, new_hprior, new_norm_ali = physics_guide.compute_energy(new_cpi, cpw, new_q)
        new_energy_grad = sum_energy(new_fc_error, new_sf_dist, new_pntr, new_hprior)
        new_energy = sum_energy(new_fc_error, new_sf_dist, new_pntr, new_hprior)
        new_grad_q, new_grad_w = torch.autograd.grad(new_energy_grad.sum(), [new_q, cpw], allow_unused=True)
        new_grad_q = new_grad_q * physics_guide.joint_mask #* disabled_joint_mask
        new_grad_q[:, 3:9] = new_grad_q[:, 3:9] * 10
        new_grad_q[:, 9:] = new_grad_q[:, 9:] * q_grad_weight
                
        with torch.no_grad():
            alpha = torch.rand(args.batch_size, device=args.device).float()
            accept = alpha < torch.exp((energy - new_energy) / temperature)
            q[accept] = new_q[accept]
            cpi[accept] = new_cpi[accept]
            energy[accept] = new_energy[accept]
            grad_q[accept] = new_grad_q[accept]
            grad_w[accept] = new_grad_w[accept]
            
            physics_guide.grad_ema_q.apply(grad_q[:, 9:] / q_grad_weight)
            
            if step % 100 == 99:
                tqdm.write(f"Step { step }, Energy: { energy.mean().detach().cpu().numpy()} "
                           + f"FC: { new_fc_error.mean().detach().cpu().numpy() } "
                           + f"PN: { new_pntr.mean().detach().cpu().numpy() } "
                           + f"SD: { (new_sf_dist / args.n_contact / n_objects).mean().detach().cpu().numpy() } "
                           + f"HP: { new_hprior.mean().detach().cpu().numpy() }")

            if args.viz and step % 1000 == 0:
                os.makedirs(os.path.join(export_dir, str(step)), exist_ok=True)
                hand_model.update_kinematics(q)
                contacts = [ hand_model.get_contact_areas(cpi[:, j]).cpu().numpy() for j in range(n_objects) ]
                contact_pts = [ hand_model.get_contact_points(cpi[:, j], cpw[:, j]).cpu().numpy() for j in range(n_objects) ]
                pntr_kpts = hand_model.get_penetration_keypoints().cpu().numpy()
                for i in range(8):
                    go.Figure([ plot_mesh(o.get_obj_mesh(i), name=f"object-{j}") for j, o in enumerate(physics_guide.object_models) ]
                    + [ plot_rect(contacts[j][i, k], name=f"contact-{j}_{k}", color='red') for j in range(n_objects) for k in range(args.n_contact) ]
                    + [ plot_mesh(tm.load("data/table.stl"), color='green'), plot_mesh(hand_model.get_meshes_from_q(None, i, True), 'lightpink', opacity=1.0) ]
                    ).write_html(os.path.join(export_dir, str(step), f"{i}.html"))

            if args.log and step % 10 == 9:
                writer.add_scalar("MALA/stepsize", step_size, step)
                writer.add_scalar("MALA/temperature", temperature, step)
                writer.add_scalar("MALA/mc_accept", accept.float().mean().detach().item(), step)
                writer.add_scalar("MALA/switch_contact", (switch_contact[0] * accept).float().mean().detach().item(), step)
                
                writer.add_scalar("Grasp/enery", energy.mean().detach().cpu().numpy(), step)
                writer.add_scalar("Grasp/fc_err", new_fc_error.mean().detach().cpu().numpy(), step)
                writer.add_scalar("Grasp/penetration", new_pntr.mean().detach().cpu().numpy(), step)
                writer.add_scalar("Grasp/distance", new_sf_dist.mean().detach().cpu().numpy(), step)
                writer.add_scalar("Grasp/hand_prior", new_hprior.mean().detach().cpu().numpy(), step)
                
    torch.cuda.empty_cache()
    
    logger.info("> Refining grasps...")
    args.sf_dist_weight = 20.0
    q = torch.tensor(q.detach().clone(), requires_grad=True)
    cpw = torch.tensor(cpw.detach().clone(), requires_grad=True)
    optimizer = torch.optim.Adam([{ "params": q, "lr": 1e-2 }, { "params": cpw, "lr": 1e-1 }])
    
    # Refine steps
    for step in trange(args.max_refine):
        optimizer.zero_grad()
        fc_error, sf_dist, pntr, hprior, norm_ali = physics_guide.compute_energy(cpi, cpw, q)
        energy = sum_energy(fc_error, sf_dist, pntr, hprior).mean()
        energy.backward()
        optimizer.step()
        
        if step % 100 == 99:
            tqdm.write(f"Step { step }, Energy: { energy.mean().detach().cpu().numpy()} "
                        + f"FC: { fc_error.mean().detach().cpu().numpy() } "
                        + f"PN: { pntr.mean().detach().cpu().numpy() } "
                        + f"SD: { (sf_dist / args.n_contact / n_objects).mean().detach().cpu().numpy() } "
                        + f"HP: { hprior.mean().detach().cpu().numpy() }")

        if args.log and step % 10 == 9:
            writer.add_scalar("Grasp/enery", energy.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar("Grasp/fc_err", fc_error.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar("Grasp/penetration", pntr.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar("Grasp/distance", sf_dist.mean().detach().cpu().numpy(), step + args.max_physics)
            writer.add_scalar("Grasp/hand_prior", hprior.mean().detach().cpu().numpy(), step + args.max_physics)
            
        if args.viz and step % 1000 == 0:
            os.makedirs(os.path.join(export_dir, str(step)), exist_ok=True)
            hand_model.update_kinematics(q)
            contacts = [ hand_model.get_contact_areas(cpi[:, j]).detach().cpu().numpy() for j in range(n_objects) ]
            contact_pts = [ hand_model.get_contact_points(cpi[:, j], cpw[:, j]).detach().cpu().numpy() for j in range(n_objects) ]
            pntr_kpts = hand_model.get_penetration_keypoints().detach().cpu().numpy()
            for i in range(8):
                go.Figure([ plot_mesh(o.get_obj_mesh(i), name=f"object-{j}") for j, o in enumerate(physics_guide.object_models) ]
                + [ plot_point_cloud(pntr_kpts[i], name=f"pntr-kpts", color='blue') ]
                + [ plot_point_cloud(contact_pts[j][i], name=f"contact-{j}") for j in range(n_objects) ]
                + [ plot_rect(contacts[j][i, k], name=f"contact-{j}_{k}", color='red') for j in range(n_objects) for k in range(args.n_contact) ]
                + [ plot_mesh(tm.load("data/table.stl"), color='green'), plot_mesh(hand_model.get_meshes_from_q(None, i, True), 'lightpink', opacity=1.0) ]
                ).write_html(os.path.join(export_dir, str(step), f"{i}.html"))
        
    logger.info("> Saving checkpoint...")
    
    fc_error, sf_dist, pntr, hprior, norm_ali = physics_guide.compute_energy(cpi, cpw, q, reduce=False)
    save(args, export_dir, uuids, physics_guide, q, cpi, cpw, fc_error, sf_dist, pntr, hprior, norm_ali)

def save(args, export_dir, uuids, physics_guide, q, cpi, cpw, fc_error, sf_dist, pntr, hprior, norm_ali, step=None):
    with torch.no_grad():
        save_dict = {
            "args": args, "uuids": uuids,
            "object_models": args.object_models,
            "q": q, "cpi": cpi, "cpw": cpw,
            "obj_scales": [obj.scale for obj in physics_guide.object_models],
            "obj_poses": [obj.get_poses() for obj in physics_guide.object_models],
            "joint_mask": physics_guide.joint_mask,
            "quantitative": {
                "fc_loss": fc_error,
                "pen": pntr,
                "surf_dist": sf_dist,
                "hand_pri": hprior,
                "norm_ali": norm_ali
            }
        }
        
        if step is None:
            torch.save(save_dict, os.path.join(export_dir, f"ckpt.pt"))
        else:
            torch.save(save_dict, os.path.join(export_dir, f"ckpt-{step}.pt"))
        
if __name__ == '__main__':
    args = parse_args()
    
    export_configs = initialize(args)

    synthesis(args, export_configs)
