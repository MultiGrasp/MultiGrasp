'''
Author: Aiden Li
Date: 2022-05-23 17:51:30
LastEditors: Aiden Li (i@aidenli.net)
LastEditTime: 2022-07-12 18:50:01
Description: v1.0
'''
import argparse
import json
import os
from ast import arg
from sched import scheduler

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from utils.ObjectModels import MeshModel
from utils.ODField import ODField
from utils.visualize_plotly import plot_mesh, plot_point_cloud_cmap, plot_point_cloud_occ


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj_name", default="bulb", type=str)
    parser.add_argument("--num_pts", default=2048, type=int)
    parser.add_argument("--obj_scale", default=1.0, type=float)
    parser.add_argument("--pts_scale", default=0.15, type=float)
    parser.add_argument("--signed", default=True, type=bool)
    parser.add_argument("--viz", default=False, type=bool)
    parser.add_argument("--n_iter", default=4000, type=int)
    parser.add_argument("--device", default='cuda', type=str)

    return parser.parse_args()

def train(args, net, mesh: MeshModel, optimizer, scheduler):
    for i in trange(args.n_iter):
        optimizer.zero_grad()

        if i % 100 == 0:
            sf_points_base = mesh._sample_pts(args.num_pts)
        if i == 2000:
            args.pts_scale *= 0.8
        sp_points = torch.randn([1, args.num_pts, 3], dtype=torch.float32, requires_grad=True, device=args.device) * args.pts_scale
        sf_points = torch.tensor(sf_points_base, dtype=torch.float32, requires_grad=True, device=args.device)
        sf_points = sf_points.clone() + torch.normal(0, 1e-2, sf_points.shape, dtype=torch.float32, device=args.device)
        points = torch.cat([sp_points, sf_points], dim=1)
        
        gt_dist, gt_grad = mesh.distance_gradient(points, signed=args.signed)
        gt_grad = torch.where(torch.isnan(gt_grad), torch.zeros_like(gt_grad, device=gt_grad.device, dtype=torch.float32), gt_grad)

        gt_field = torch.cat([gt_dist.squeeze().unsqueeze(-1), gt_grad.squeeze()], dim=-1).detach()

        pred_field = net(points)
        mserr = ((pred_field - gt_field)**2).mean()
        
        pred_grad = pred_field[..., 1:]
        nmerr = (pred_grad.norm(dim=-1) - 1).abs().mean()
        
        pred_grad = F.normalize(pred_grad, dim=-1)
        alerr = - (gt_grad * pred_grad).mean()
        loss = 10. * mserr + 5. * alerr + 1 * nmerr
        loss.backward()

        optimizer.step()
        scheduler.step()

        if i % 250 == 0:
            with torch.no_grad():
                pred_grad = pred_field[..., 1:]
                pred_grad = pred_grad / pred_grad.norm(dim=-1, keepdim=True)
                mserr_crit = torch.abs(pred_field - gt_field).mean()
                align_crit = (gt_grad * pred_grad).sum(-1).mean()
                print(f"Iteration {i}  Dist MSError = {mserr_crit.item():.8f} Mean Ali  = {align_crit.item():.8f} Norm Error = {nmerr.item():.8f}")

    return net, mserr_crit.item(), align_crit.item()


def eval(args, net, mesh: MeshModel):
    net.eval()
    sf_points_base = mesh._sample_pts(args.num_pts)
    sp_points = torch.randn([1, args.num_pts, 3], dtype=torch.float32, requires_grad=True, device=args.device) * args.pts_scale
    sf_points = torch.tensor(sf_points_base, dtype=torch.float32, requires_grad=True, device=args.device)
    sf_points = sf_points.clone() + torch.normal(0, 1e-2, sf_points.shape, dtype=torch.float32, device=args.device)
    points = torch.cat([sp_points, sf_points], dim=1)

    gt_dist, gt_grad = mesh.distance_gradient(points, signed=args.signed)
    gt_dist = gt_dist.squeeze()
    gt_grad = gt_grad.squeeze()

    points.detach()
    pred_field = net(points).squeeze()
    pred_dist = pred_field[:, 0]
    pred_grad = pred_field[:, 1:]
    pred_grad = pred_grad / (torch.norm(pred_grad, dim=-1, keepdim=True) + 1e-12)

    dist_error = torch.abs(gt_dist - pred_dist)
    grad_align = (gt_grad * pred_grad).sum(-1)

    points = points.detach().cpu()
    dist_error = dist_error.detach().cpu()
    grad_align = grad_align.detach().cpu()

    print(f"[Dist pred] Mean: { dist_error.mean()}, Max { dist_error.max() }, Min { dist_error.min() }")
    print(f"[Grad pred] Mean: { grad_align.mean()}, Max { grad_align.max() }, Min { grad_align.min() }")

    if args.viz:
        export_basedir = os.path.join("demo", "ODFields")
        os.makedirs(export_basedir, exist_ok=True)
        go.Figure(
            [
                plot_mesh(mesh.get_obj_mesh(), opacity=0.8),
                plot_point_cloud_cmap(points[0].detach().cpu(), color_levels=pred_dist.detach().cpu())
            ]
        ).write_html(os.path.join(export_basedir, f"{args.obj_name}.html"))

if __name__ == '__main__':
    args = get_args()

    print(f"Training ODF for {args.obj_name}")
    mesh = MeshModel(batch_size=1, object_model=args.obj_name, obj_scale=args.obj_scale, device=args.device)
    field = ODField().to(args.device)

    optimizer = torch.optim.Adam(field.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

    res = train(args, field, mesh, optimizer, scheduler)
    if res is None:
        print()
        exit()

    field, mserr_crit, align_crit = res
    eval(args, field, mesh)

    os.makedirs(mesh.data_dir, exist_ok=True)
    sd_export_path = os.path.join(mesh.data_dir, "sdfield.pt" if args.signed else "udfield.pt")
    torch.save(field.state_dict(), sd_export_path)
