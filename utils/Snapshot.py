'''
LastEditTime: 2022-07-02 15:10:18
Description: Snapshot system, with cpw for robotic hands
Date: 2021-10-28 05:01:01
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''

from operator import concat
from posixpath import join

from utils.HandModel import get_hand_model
from utils.ObjectModels import get_object_model
from utils.PhysicsGuide import PhysicsGuide
from utils.visualize_plotly import *


def load_from_snapshot(args=None, path=None, batch_rec=True):
    if args is not None:
        path = args.checkpoint
        
    load = torch.load(path)
    loaded_args = load['args']
    
    hand_model = get_hand_model(loaded_args.hand_model, batch_size=loaded_args.batch_size, device=loaded_args.device)
    physics_guide = PhysicsGuide(hand_model, loaded_args)
    
    q = load['q']
    cpi = load['cpi']
    cpw = load['cpw']
    
    for ind, gr_obj in enumerate(loaded_args.object_models):
        transl, orient = load['obj_poses'][ind]
        obj_scale = load['obj_scales'][ind]
        
        obj_model = get_object_model(gr_obj, loaded_args.batch_size, scale=obj_scale, device=loaded_args.device)
        transl = transl.to(loaded_args.device)
        orient = orient.to(loaded_args.device)
        obj_model.update_pose(transl, orient)
        if gr_obj == "sphere":
            obj_model.radius = obj_scale
        else:
            obj_model.scale = obj_scale
        
        physics_guide.append_object(obj_model)
        
    uuids = load['uuids']
    quantitative = load['quantitative']

    return uuids, loaded_args, q, cpi, cpw, physics_guide, hand_model, quantitative
        