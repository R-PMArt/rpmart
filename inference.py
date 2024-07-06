from typing import Tuple, List, Optional
from omegaconf import OmegaConf
import os
import tqdm
import itertools
from itertools import combinations
import random
import numpy as np
import torch
import torch.nn as nn
import cupy as cp
# import cudf
# import cuml
from sklearn.cluster import KMeans
# from cuml.cluster import DBSCAN
# from cuml.common.device_selection import using_device_type
import MinkowskiEngine as ME
import open3d as o3d

from models.roartnet import create_shot_encoder, create_encoder
from models.voting import ppf_kernel, rot_voting_kernel, ppf4d_kernel
from utilities.metrics_utils import calc_translation_error, calc_direction_error
from utilities.vis_utils import visualize, visualize_translation_voting, visualize_rotation_voting, visualize_confidence_voting
from utilities.data_utils import pc_normalize, farthest_point_sample, fibonacci_sphere
from utilities.env_utils import setup_seed
from utilities.constants import seed, light_blue_color, red_color, dark_red_color, dark_green_color, yellow_color
from src_shot.build import shot


def voting_translation(pc:np.ndarray, tr_offsets:np.ndarray, point_idxs:np.ndarray, confs:np.ndarray, 
                       resolution:float, voting_num:int, device:int, 
                       translation2pc:bool, multi_candidate:bool, candidate_threshold:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pc: (N, 3), tr_offsets: (N_t, 2), point_idxs: (N_t, 2), confs: (N_t,)
    block_size = (tr_offsets.shape[0] + 512 - 1) // 512
    pc_min = np.min(pc, 0)
    pc_max = np.max(pc, 0)
    corner_min = pc_min - (pc_max - pc_min)
    corner_max = pc_max + (pc_max - pc_min)
    corners = np.stack([corner_min, corner_max])
    grid_res = ((corners[1] - corners[0]) / resolution).astype(np.int32) + 1

    with cp.cuda.Device(device):
        grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
        
        ppf_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.ascontiguousarray(cp.asarray(pc).astype(cp.float32)),
                cp.ascontiguousarray(cp.asarray(tr_offsets).astype(cp.float32)),
                cp.ascontiguousarray(cp.asarray(confs).astype(cp.float32)),
                cp.ascontiguousarray(cp.asarray(point_idxs).astype(cp.int32)),
                grid_obj,
                cp.ascontiguousarray(cp.asarray(corners[0]).astype(cp.float32)),
                cp.float32(resolution),
                cp.int32(tr_offsets.shape[0]),
                cp.int32(voting_num),
                cp.int32(grid_obj.shape[0]),
                cp.int32(grid_obj.shape[1]),
                cp.int32(grid_obj.shape[2])
            )
        )
    
        if not multi_candidate:
            cand = cp.array(cp.unravel_index(cp.array([cp.argmax(grid_obj, axis=None)]), grid_obj.shape)).T[::-1]
            cand_world = cp.asarray(corners[0]) + cand * resolution
        else:
            indices = cp.indices(grid_obj.shape)
            indices_list = cp.transpose(indices, (1, 2, 3, 0)).reshape(-1, len(grid_obj.shape))
            votes_list = grid_obj.reshape(-1)
            grid_pc = cp.asarray(corners[0]) + indices_list * resolution
            normalized_votes_list = votes_list / cp.max(votes_list)
            candidates = grid_pc[normalized_votes_list >= candidate_threshold]
            candidate_weights = normalized_votes_list[normalized_votes_list >= candidate_threshold]
            candidate_weights = candidate_weights / cp.sum(candidate_weights)
            cand_world = cp.sum(candidates * candidate_weights[:, None], axis=0)[None, :]

        if translation2pc:
            pc_cp = cp.asarray(pc)
            best_idx = cp.linalg.norm(pc_cp - cand_world, axis=-1).argmin()
            translation = pc_cp[best_idx]
        else:
            translation = cand_world[0]
    
    return (translation.get(), grid_obj.get(), corners)

def voting_rotation(pc:np.ndarray, rot_offsets:np.ndarray, point_idxs:np.ndarray, confs:np.ndarray, 
                    rot_candidate_num:int, angle_tol:float, voting_num:int, bmm_size:int, device:int, 
                    multi_candidate:bool, candidate_threshold:float, rotation_cluster:bool, kmeans:Optional[KMeans], 
                    rotation_multi_neighbor:bool, neighbor_threshold:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pc: (N, 3), rot_offsets: (N_t,), point_idxs: (N_t, 2), confs: (N_t,)
    block_size = (rot_offsets.shape[0] + 512 - 1) // 512
    sphere_pts = np.array(fibonacci_sphere(rot_candidate_num))
    expanded_confs = confs[:, None].repeat(voting_num, axis=-1).reshape(-1, 1)

    with cp.cuda.Device(device):
        candidates = cp.zeros((rot_offsets.shape[0], voting_num, 3), cp.float32)

        rot_voting_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.ascontiguousarray(cp.asarray(pc).astype(cp.float32)),
                cp.ascontiguousarray(cp.asarray(rot_offsets).astype(cp.float32)),
                candidates,
                cp.ascontiguousarray(cp.asarray(point_idxs).astype(cp.int32)),
                cp.int32(rot_offsets.shape[0]),
                cp.int32(voting_num)
            )
        )
        
        candidates = candidates.get().reshape(-1, 3)
    
    with torch.no_grad():
        candidates = torch.from_numpy(candidates).cuda(device)
        expanded_confs = torch.from_numpy(expanded_confs).cuda(device)
        sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda(device)
        counts = torch.zeros((sphere_pts.shape[0],), dtype=torch.float32).cuda(device)  # (rot_candidate_num,)

        for i in range((candidates.shape[0] - 1) // bmm_size + 1):
            cos = candidates[i * bmm_size:(i + 1) * bmm_size].mm(sph_cp)        # (bmm_size, rot_candidate_num)
            if not rotation_multi_neighbor:
                voting = (cos > np.cos(2 * angle_tol / 180 * np.pi)).float()    # (bmm_size, rot_candidate_num)
            else:
                # voting_indices = torch.topk(cos, neighbors_num, dim=-1)[1]
                # voting_mask = torch.zeros_like(cos)
                # voting_mask.scatter_(1, voting_indices, 1)
                voting_mask = (cos > np.cos(neighbor_threshold / 180 * np.pi)).float()
                voting = cos * voting_mask                                      # (bmm_size, rot_candidate_num)
            counts += torch.sum(voting * expanded_confs[i * bmm_size:(i + 1) * bmm_size], dim=0)

    if not multi_candidate:
        direction = np.array(sphere_pts[np.argmax(counts.cpu().numpy())])
    else:
        counts_list = counts.cpu().numpy()
        normalized_counts_list = counts_list / np.max(counts_list)
        candidates = sphere_pts[normalized_counts_list >= candidate_threshold]
        candidate_weights = normalized_counts_list[normalized_counts_list >= candidate_threshold]
        candidate_weights = candidate_weights / np.sum(candidate_weights)
        if not rotation_cluster:
            direction = np.sum(candidates * candidate_weights[:, None], axis=0)
            direction /= np.linalg.norm(direction)
        else:
            if candidates.shape[0] == 1:
                direction = candidates[0]
            else:
                kmeans.fit(candidates)
                candidate_center1 = kmeans.cluster_centers_[0]
                candidate_center2 = kmeans.cluster_centers_[1]
                cluster_cos_theta = np.dot(candidate_center1, candidate_center2)
                cluster_cos_theta = np.clip(cluster_cos_theta, -1., 1.)
                cluster_theta = np.arccos(cluster_cos_theta)
                if cluster_theta > np.pi/2:
                    candidate_clusters = kmeans.labels_
                    clusters_num = np.bincount(candidate_clusters)
                    if clusters_num[0] == clusters_num[1]:
                        candidate_weights1 = candidate_weights[candidate_clusters == 0]
                        candidate_weights2 = candidate_weights[candidate_clusters == 1]
                        if np.sum(candidate_weights1) >= np.sum(candidate_weights2):
                            candidates = candidates[candidate_clusters == 0]
                            candidate_weights = candidate_weights[candidate_clusters == 0]
                            candidate_weights = candidate_weights / np.sum(candidate_weights)
                            direction = np.sum(candidates * candidate_weights[:, None], axis=0)
                            direction /= np.linalg.norm(direction)
                        else:
                            candidates = candidates[candidate_clusters == 1]
                            candidate_weights = candidate_weights[candidate_clusters == 1]
                            candidate_weights = candidate_weights / np.sum(candidate_weights)
                            direction = np.sum(candidates * candidate_weights[:, None], axis=0)
                            direction /= np.linalg.norm(direction)
                    else:
                        max_cluster = np.bincount(candidate_clusters).argmax()
                        candidates = candidates[candidate_clusters == max_cluster]
                        candidate_weights = candidate_weights[candidate_clusters == max_cluster]
                        candidate_weights = candidate_weights / np.sum(candidate_weights)
                        direction = np.sum(candidates * candidate_weights[:, None], axis=0)
                        direction /= np.linalg.norm(direction)
                else:
                    direction = np.sum(candidates * candidate_weights[:, None], axis=0)
                    direction /= np.linalg.norm(direction)

    return (direction, sphere_pts, counts.cpu().numpy())


def inference_fn(pc:np.ndarray, pc_color:Optional[np.ndarray], shot_encoder:nn.Module, encoder:nn.Module, 
                 denoise:bool, normalize:str, resolution:float, receptive_field:int, sample_points_num:int, sample_tuples_num:int, tuple_more_num:int, 
                 voting_num:int, rot_bin_num:int, angle_tol:float, 
                 translation2pc:bool, multi_candidate:bool, candidate_threshold:float, rotation_cluster:bool, 
                 rotation_multi_neighbor:bool, neighbor_threshold:float, bmm_size:int, joint_num:int, device:int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if not hasattr(inference_fn, 'permutations'):
        inference_fn.permutations = list(itertools.permutations(range(sample_points_num), 2))
        inference_fn.sample_points_num = sample_points_num
    else:
        if inference_fn.sample_points_num != sample_points_num:
            inference_fn.permutations = list(itertools.permutations(range(sample_points_num), 2))
            inference_fn.sample_points_num = sample_points_num
        else:
            pass
    if rotation_cluster:
        # kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto')
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
    else:
        kmeans = None
    rot_candidate_num = int(4 * np.pi / (angle_tol / 180 * np.pi))
    has_rgb = pc_color is not None

    # preprocess point cloud
    if denoise:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        _, index = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
        pc = pc[index]
        if has_rgb:
            pc_color = pc_color[index]
    
    pc, center, scale = pc_normalize(pc, normalize)

    indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=resolution)[1]
    pc = np.ascontiguousarray(pc[indices].astype(np.float32))
    if has_rgb:
        pc_color = pc_color[indices]
    
    pc_normal = shot.estimate_normal(pc, resolution * receptive_field).reshape(-1, 3).astype(np.float32)
    pc_normal[~np.isfinite(pc_normal)] = 0

    pc_shot = shot.compute(pc, resolution * receptive_field, resolution * receptive_field).reshape(-1, 352).astype(np.float32)
    pc_shot[~np.isfinite(pc_shot)] = 0

    pc, indices = farthest_point_sample(pc, sample_points_num)
    pc_normal = pc_normal[indices]
    pc_shot = pc_shot[indices]
    if has_rgb:
        pc_color = pc_color[indices]
    
    point_idxs = random.sample(inference_fn.permutations, sample_tuples_num)
    point_idxs = np.array(point_idxs, dtype=np.int64)
    point_idxs_more = np.random.randint(0, sample_points_num, size=(sample_tuples_num, tuple_more_num), dtype=np.int64)
    point_idxs_all = np.concatenate([point_idxs, point_idxs_more], axis=-1)

    pcs = torch.from_numpy(pc)[None, ...].cuda(device)
    pc_normals = torch.from_numpy(pc_normal)[None, ...].cuda(device)
    pc_shots = torch.from_numpy(pc_shot)[None, ...].cuda(device)
    if has_rgb:
        pc_colors = torch.from_numpy(pc_color)[None, ...].cuda(device)
    point_idxs_alls = torch.from_numpy(point_idxs_all)[None, ...].cuda(device)

    # inference
    with torch.no_grad():
        shot_feat = shot_encoder(pc_shots)      # (1, N, N_s)

        shot_inputs = torch.cat([
            torch.gather(shot_feat, 1, 
                        point_idxs_alls[:, :, i:i+1].expand(
                        (1, sample_tuples_num, shot_feat.shape[-1]))) 
            for i in range(point_idxs_alls.shape[-1])], dim=-1)     # (1, N_t, N_s * (2 + N_m))
        normal_inputs = torch.cat([torch.max(
            torch.sum(torch.gather(pc_normals, 1, 
                                point_idxs_alls[:, :, i:i+1].expand(
                                (1, sample_tuples_num, pc_normals.shape[-1]))) * 
                    torch.gather(pc_normals, 1, 
                                point_idxs_alls[:, :, j:j+1].expand(
                                (1, sample_tuples_num, pc_normals.shape[-1]))), 
                    dim=-1, keepdim=True), 
            torch.sum(-torch.gather(pc_normals, 1, 
                                point_idxs_alls[:, :, i:i+1].expand(
                                (1, sample_tuples_num, pc_normals.shape[-1]))) * 
                    torch.gather(pc_normals, 1, 
                                point_idxs_alls[:, :, j:j+1].expand(
                                (1, sample_tuples_num, pc_normals.shape[-1]))), 
                    dim=-1, keepdim=True)) 
            for (i, j) in combinations(np.arange(point_idxs_alls.shape[-1]), 2)], dim=-1)   # (1, N_t, (2+N_m \choose 2))
        coord_inputs = torch.cat([
            torch.gather(pcs, 1, 
                        point_idxs_alls[:, :, i:i+1].expand(
                        (1, sample_tuples_num, pcs.shape[-1]))) - 
            torch.gather(pcs, 1, 
                        point_idxs_alls[:, :, j:j+1].expand(
                        (1, sample_tuples_num, pcs.shape[-1]))) 
            for (i, j) in combinations(np.arange(point_idxs_alls.shape[-1]), 2)], dim=-1)   # (1, N_t, 3 * (2+N_m \choose 2))
        if has_rgb:
            rgb_inputs = torch.cat([
                    torch.gather(pc_colors, 1, 
                                    point_idxs_alls[:, :, i:i+1].expand(
                                    (1, sample_tuples_num, pc_colors.shape[-1]))) 
                    for i in range(point_idxs_alls.shape[-1])], dim=-1)     # (1, N_t, 3 * (2 + N_m))
            inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs, rgb_inputs], dim=-1)
        else:
            inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], dim=-1)
        preds = encoder(inputs)                             # (1, N_t, (2 + N_r + 2 + 1) * J)
        pred = preds.cpu().numpy().astype(np.float32)[0]    # (N_t, (2 + N_r + 2 + 1) * J)
        pred_tensor = torch.from_numpy(pred)

        pred_translations, pred_rotations, pred_affordances = [], [], []
        for j in range(joint_num):
            # conf selection
            pred_conf = torch.sigmoid(pred_tensor[:, -1*joint_num+j])                   # (N_t,)
            not_selected_indices = pred_conf < 0.5
            pred_conf[not_selected_indices] = 0
            # pred_conf[pred_conf > 0] = 1
            pred_conf = pred_conf.numpy()

            # translation voting
            pred_tr = pred[:, 2*j:2*(j+1)]                                              # (N_t, 2)
            pred_translation, grid_obj, corners = voting_translation(pc, pred_tr, point_idxs, pred_conf, 
                                                                     resolution, voting_num, device, 
                                                                     translation2pc, multi_candidate, candidate_threshold)
            pred_translations.append(pred_translation)
            
            # rotation voting
            pred_rot = pred_tensor[:, (2*joint_num+rot_bin_num*j):(2*joint_num+rot_bin_num*(j+1))]  # (N_t, rot_bin_num)
            pred_rot = torch.softmax(pred_rot, dim=-1)
            pred_rot = torch.multinomial(pred_rot, 1).float()[:, 0]                     # (N_t,)
            pred_rot = pred_rot / (rot_bin_num - 1) * np.pi
            pred_rot = pred_rot.numpy()
            pred_direction, sphere_pts, counts = voting_rotation(pc, pred_rot, point_idxs, pred_conf, 
                                                                 rot_candidate_num, angle_tol, voting_num, bmm_size, device, 
                                                                 multi_candidate, candidate_threshold, rotation_cluster, kmeans, 
                                                                 rotation_multi_neighbor, neighbor_threshold)
            pred_rotations.append(pred_direction)

            # affordance voting
            pred_afford = pred[:, (2*joint_num+rot_bin_num*joint_num+2*j):(2*joint_num+rot_bin_num*joint_num+2*(j+1))]  # (N_t, 2)
            pred_affordance, agrid_obj, acorners = voting_translation(pc, pred_afford, point_idxs, pred_conf, 
                                                                      resolution, voting_num, device, 
                                                                      translation2pc, multi_candidate, candidate_threshold)
            pred_affordances.append(pred_affordance)
    return (pred_translations, pred_rotations, pred_affordances)


if __name__ == '__main__':
    pass
