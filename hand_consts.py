import numpy as np
import torch
import torch.nn.functional as F

# 00: 'palm'
# 01: 'palm'
# 02: 'ffproximal'
# 03: 'ffmiddle'
# 04: 'ffdistal'
# 05: 'mfproximal'
# 06: 'mfmiddle'
# 07: 'mfdistal'
# 08: 'rfproximal'
# 09: 'rfmiddle'
# 10: 'rfdistal'
# 11: 'lfproximal'
# 12: 'lfmiddle'
# 13: 'lfdistal'
# 14: 'thproximal'
# 15: 'thproximal'
# 16: 'thmiddle'
# 17: 'thmiddle'
# 18: 'thdistal'

# 'WRJ2'
# 'WRJ1'
# 00: 'FFJ4'
# 01: 'FFJ3'
# 02: 'FFJ2'
# 03: 'FFJ1'
# 04: 'MFJ4'
# 05: 'MFJ3'
# 06: 'MFJ2'
# 07: 'MFJ1'
# 08: 'RFJ4'
# 09: 'RFJ3'
# 10: 'RFJ2'
# 11: 'RFJ1'
# 12: 'LFJ5'
# 13: 'LFJ4'
# 14: 'LFJ3'
# 15: 'LFJ2'
# 16: 'LFJ1'
# 17: 'THJ5'
# 18: 'THJ4'
# 19: 'THJ3'
# 20: 'THJ2'
# 21: 'THJ1'
# 22: 
# 23: 


def get_contact_pool(contact):
    idxs_pool = []
    
    # if 0 in contact: # Palm
    #     idxs_pool += np.arange(0, 2).tolist()
    # if 1 in contact: # Thumb
    #     idxs_pool += np.arange(22, 25).tolist()
    # if 2 in contact: # Index
    #     idxs_pool += np.arange(2, 7).tolist()
    # if 3 in contact: # Middle
    #     idxs_pool += np.arange(7, 12).tolist()
    # if 4 in contact: # Ring
    #     idxs_pool += np.arange(12, 17).tolist()
    # if 5 in contact: # Pinky
    #     idxs_pool += np.arange(17, 22).tolist()
        
    if 0 in contact: # Palm
        idxs_pool += np.arange(0, 2).tolist()
    if 1 in contact: # Thumb
        idxs_pool += np.arange(14, 17).tolist()
    if 2 in contact: # Index
        idxs_pool += np.arange(2, 5).tolist()
    if 3 in contact: # Middle
        idxs_pool += np.arange(5, 8).tolist()
    if 4 in contact: # Ring
        idxs_pool += np.arange(8, 11).tolist()
    if 5 in contact: # Pinky
        idxs_pool += np.arange(11, 14).tolist()
        
    return idxs_pool


def contact_joint_mask(fingers, contact):

    idxs_pool = []
    joint_mask = np.zeros(22, dtype=float)
    # joint_mask = np.zeros(24, dtype=float)
    
    if 0 in contact: # Palm
        idxs_pool += [0, 1]
        
    if 1 in contact: # Thumb
        idxs_pool += [14, 15, 16, 17, 18]
    if 1 in fingers: # Thumb
        joint_mask[17:22] = 1.0
        
    if 2 in contact: # Index
        idxs_pool += [2, 3, 4]
    if 2 in fingers: # Index
        joint_mask[0:4] = 1.0
        
    if 3 in contact: # Middle
        idxs_pool += [5, 6, 7]
    if 3 in fingers: # Middle
        joint_mask[4:8] = 1.0
        
    if 4 in contact: # Ring
        idxs_pool += [8, 9, 10]
    if 4 in fingers: # Ring
        joint_mask[8:12] = 1.0
        
    if 5 in contact: # Pinky
        idxs_pool += [11, 12, 13]
    if 5 in fingers: # Pinky
        joint_mask[12:17] = 1.0
    
    # joint_mask[:2] = 0.0 if fix_knuckle else 1.0
    
    joint_mask = np.pad(joint_mask, (9, 0), 'constant', constant_values=1.0)
    
    return idxs_pool, joint_mask


def joint_mask_sd(contact_idxs, fix_knuckle=True):
    B, N_c = contact_idxs.shape
    idxs_pool = []
    joint_mask = torch.ones(B, 22, dtype=torch.float32, device=contact_idxs.device)
    
    cid_to_finger = [
        [], [],                                                                         # Palm
        [2, 3],   [2, 3, 4],   [2, 3, 4, 5],                                            # Index
        [6, 7],   [6, 7, 8],   [6, 7, 8, 9],                                            # Middle
        [10, 11], [10, 11, 12], [10, 11, 12, 13],                                       # Ring
        [15, 16, 17], [14, 15, 16, 17], [14, 15, 16, 17, 18],                           # Pinky
        [19, 20], [19, 20], [19, 20, 21, 22], [19, 20, 21, 22], [19, 20, 21, 22, 23]    # Thumb
    ]   
    
    cid_to_finger = [ (np.array(l) - 2).tolist() for l in cid_to_finger ]
    
    for cid, jid in enumerate(cid_to_finger):
        if len(jid) == 0:
            continue
        to_fix = torch.zeros(B, dtype=torch.bool, device=contact_idxs.device)
        for i_c in range(N_c):
            to_fix = to_fix + (contact_idxs[:, i_c] == cid)
        for j in jid:
            joint_mask[torch.where(to_fix)[0], j] = 0.0
        
    joint_mask = F.pad(joint_mask, (9, 0), value=1.0)
        
    return joint_mask

contact_groups = [
    [ [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5] ],
    [ [0, 1, 2, 3], [0, 3, 4, 5] ],
    [ [0, 1, 2], [0, 2, 3, 4, 5] ],
    [ [0, 1, 5], [0, 2, 3, 4] ],
]