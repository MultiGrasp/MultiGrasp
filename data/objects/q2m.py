from collections import defaultdict
import numpy as np
from pytorch3d import transforms
import torch
import json
from icecream import ic

rotmats = {}

rots = json.load(open("drop_rot_filtered_new.json", "r"))
for k, v in rots.items():
    quaternions = torch.tensor([ r[1] for r in v ])
    quaternions = torch.cat([quaternions[:, 3:4], quaternions[:, 0:3]], dim=-1)
    mats = transforms.quaternion_to_matrix(quaternions)
    rotmats[k] = mats.numpy().tolist()
    
json.dump(rotmats, open("rotations_filtered_new.json", 'w'))
