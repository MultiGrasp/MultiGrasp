'''
LastEditTime: 2022-02-15 08:28:42
Description: More utilities
Date: 2021-10-28 05:01:01
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
import torch

def shuffle_tensor(tensor, dim=-1):
    return tensor[:, torch.randperm(tensor.shape[dim])].view(tensor.size())

def accept_and_tile(tensor, accept_indices, trim=False):
    """Accept parts of a tensor and retile them to recover (may outnumbers by a bit) the batch size.
    """
    old_batch_size = tensor.shape[0]
    tile_factor = [int(tensor.shape[0] / accept_indices.shape[0])] + [1] * (tensor.dim() - 1)
    
    if trim:
        return tensor[accept_indices].tile(tile_factor)[:old_batch_size]
    else:
        return tensor[accept_indices].tile(tile_factor)

def tensor_in_arr(ten, dic):
    for elem in dic:
        if ten.equal(elem):
            return True
    return False

def neighbors_on_mesh(ind, tri=None, neighbor_array=None):
    if tri is None:
        return neighbor_array[ind]
    elif neighbor_array is None:
        neighbors = []
        for t in tri:
            if ind == t[0]:
                if t[1] not in neighbors:
                    neighbors.append(t[1])
                if t[2] not in neighbors:
                    neighbors.append(t[2])
            if ind == t[1]:
                if t[0] not in neighbors:
                    neighbors.append(t[0])
                if t[2] not in neighbors:
                    neighbors.append(t[2])
            if ind == t[2]:
                if t[0] not in neighbors:
                    neighbors.append(t[0])
                if t[1] not in neighbors:
                    neighbors.append(t[1])
    else:
        raise NotImplementedError()
    return neighbors


def fps_indices(dist_matrix, n_sample, init_i=0):
    """
    Args:
        dist_matrix: N x N
    """
    N = dist_matrix.shape[0]
    selected = torch.zeros([dist_matrix.shape[0]], device=dist_matrix.device, dtype=torch.int)
    selected[init_i] = 1
    
    while n_sample > 1:
        selected_exp = selected.unsqueeze(0).expand([N, N])
        dist_to_selected = (dist_matrix * selected_exp).sum(dim=1)
        unselected_dist_to_selected = dist_to_selected * (1 - selected) / selected.sum()
        
        max_dist = torch.max(unselected_dist_to_selected).item()
        next_target = torch.argmax(unselected_dist_to_selected).item()
        
        selected[next_target] = 1
        n_sample -= 1
        
        print(f"Selected { next_target } with dist { max_dist }")
        
    return torch.where(selected == 1)[0]
