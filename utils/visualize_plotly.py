'''
LastEditTime: 2022-07-02 11:51:15
Description: Your description
Date: 2021-11-04 04:54:29
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
import io
import os
import numpy as np
import torch
import torch
import trimesh as tm
from plotly import graph_objects as go
from PIL import Image

colors = [
    'blue', 'red', 'yellow', 'pink', 'gray', 'orange'
]

def plot_connection(x, y, color='white', name='conn'):
    return [
        go.Scatter3d(
            x=[x[i, 0], y[i, 0]],
            y=[x[i, 1], y[i, 1]],
            z=[x[i, 2], y[i, 2]],
            mode='lines',
            line={
                'color': color,
                'width': 2
            }
        ) for  i in range(x.shape[0])
    ]

def plot_mesh(mesh, color='lightblue', opacity=1.0, name='mesh'):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True)

def plot_hand(verts, faces, color='lightpink', opacity=1.0):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color, opacity=opacity, showlegend=True)

def plot_contact_points(pts, grad, color='lightpink'):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], u=-grad[:, 0], v=-grad[:, 1], w=-grad[:, 2], anchor='tip',
                   colorscale=[(0, color), (1, color)], sizemode='absolute', sizeref=0.2, opacity=0.5, showlegend=True)

def plot_point_cloud(pts, **kwargs):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        **kwargs
    )
    
def plot_rect(verts, color='lightblue', opacity=1.0, name='mesh'):
    # plot a rectangle with verts as vertices
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=[0, 0],
        j=[1, 3],
        k=[2, 2],
        color=color, opacity=opacity, name=name, showlegend=True)


occ_cmap = lambda levels: [f"rgb({int(255)},{int(255)},{int(255)})" if x <= 1e-12 else f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]

def plot_point_cloud_occ(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': occ_cmap(color_levels),
            'size': 3,
            'opacity': 1
        }
    )
    
def plot_point_cloud_cmap(pts, color_levels=None, name="points"):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        name=name,
        mode='markers',
        marker={
            'color': color_levels,
            'size': 3,
            'opacity': 1
        }
    )

def plot_grasps(directory, uuids, physics_guide, handcodes, contact_idxs, ret_plots=False, save_html=True, include_contacts=True, indices=None, export_mesh=False):
    object_models = physics_guide.object_models
    os.makedirs(directory, exist_ok=True)
    
    if indices is None:
        indices = np.arange(len(uuids)).tolist()
        
    if include_contacts:
        contact_points = []
        for i in range(contact_idxs.shape[1]):
            contact_points.append(physics_guide.get_contacts(contact_idxs[:, i], handcodes)[0])
        contact_points = torch.cat(contact_points, dim=1).detach().cpu()
    
    plots = []

    for i, batch_idx in enumerate(indices):
        to_plot = []

        to_plot += physics_guide.hand_model.get_plotly_data(handcodes[batch_idx:batch_idx+1, -1])

        for obj_ind, obj in enumerate(object_models):
            obj_plot = obj.get_plot(batch_idx)
            obj_mesh = obj.get_obj_mesh(batch_idx)
            to_plot.append(obj.get_plot(batch_idx))
            if include_contacts:
                to_plot.append(plot_point_cloud(contact_points[batch_idx], color=colors[obj_ind]))
        
        fig = go.Figure(to_plot)
        
        if export_mesh:
            hand_mesh = physics_guide.hand_model.get_meshes_from_q(handcodes[batch_idx:batch_idx+1, -1])
            hand_mesh = tm.util.concatenate(hand_mesh)
            hand_mesh.export(os.path.join(f"{ directory }", f"h_{ str(uuids[i]) }.stl"))
            obj_mesh.export(os.path.join(f"{ directory }", f"o_{ str(uuids[i]) }.stl"))
        if save_html:
            fig.write_html(os.path.join(f"{ directory }", f"{ batch_idx }_{ str(uuids[i]) }.html"))
        if ret_plots:
            plots.append(torch.from_numpy(np.asarray(Image.open(io.BytesIO(fig.to_image(format="png", width=1280, height=720))))))
            
    if ret_plots:
        return plots
   
