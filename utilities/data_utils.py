from typing import List, Tuple, Union, Dict, Optional
import math
import numpy as np
from scipy.spatial.transform import Rotation as srot
from scipy.optimize import least_squares, linear_sum_assignment
import torch
import xml.etree.ElementTree as ET

from .metrics_utils import calc_translation_error_batch, calc_direction_error_batch


def read_joints_from_urdf_file(urdf_file):
    tree_urdf = ET.parse(urdf_file)
    root_urdf = tree_urdf.getroot()

    joint_dict = {}
    for joint in root_urdf.iter('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        for child in joint.iter('child'):
            joint_child = child.attrib['link']
        for parent in joint.iter('parent'):
            joint_parent = parent.attrib['link']
        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                joint_xyz = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                joint_xyz = [0, 0, 0]
            if 'rpy' in origin.attrib:
                joint_rpy = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                joint_rpy = [0, 0, 0]
        if joint_type == 'prismatic' or joint_type == 'revolute' or joint_type == 'continuous':
            for axis in joint.iter('axis'):
                joint_axis = [float(x) for x in axis.attrib['xyz'].split()]
        else:
            joint_axis = None
        if joint_type == 'prismatic' or joint_type == 'revolute':
            for limit in joint.iter('limit'):
                joint_limit = [float(limit.attrib['lower']), float(limit.attrib['upper'])]
        else:
            joint_limit = None

        joint_dict[joint_name] = {
            'type': joint_type,
            'parent': joint_parent,
            'child': joint_child,
            'xyz': joint_xyz,
            'rpy': joint_rpy,
            'axis': joint_axis,
            'limit': joint_limit
        }

    return joint_dict


def pc_normalize(pc:np.ndarray, normalize_method:str) -> Tuple[np.ndarray, np.ndarray, float]:
    if normalize_method == 'none':
        pc_normalized = pc
        center = np.array([0., 0., 0.]).astype(pc.dtype)
        scale = 1.
    elif normalize_method == 'mean':
        center = np.mean(pc, axis=0)
        pc_normalized = pc - center
        scale = np.max(np.sqrt(np.sum(pc_normalized ** 2, axis=1)))
        pc_normalized = pc_normalized / scale
    elif normalize_method == 'bound':
        center = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2
        pc_normalized = pc - center
        scale = np.max(np.sqrt(np.sum(pc_normalized ** 2, axis=1)))
        pc_normalized = pc_normalized / scale
    elif normalize_method == 'median':
        center = np.median(pc, axis=0)
        pc_normalized = pc - center
        scale = np.max(np.sqrt(np.sum(pc_normalized ** 2, axis=1)))
        pc_normalized = pc_normalized / scale
    else:
        raise NotImplementedError

    return (pc_normalized, center, scale)

def joints_normalize(joint_translations:Optional[np.ndarray], joint_rotations:Optional[np.ndarray], center:np.ndarray, scale:float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if joint_translations is None and joint_rotations is None:
        return (None, None)
    
    J = joint_translations.shape[0] if joint_translations is not None else joint_rotations.shape[0]
    joint_translations_normalized, joint_rotations_normalized = [], []
    for j in range(J):
        if joint_translations is not None:
            joint_translation_normalized = (joint_translations[j] - center) / scale
            joint_translations_normalized.append(joint_translation_normalized)
        if joint_rotations is not None:
            # joint_axis_normalized = (joint_rotations[j] - center) / scal
            # joint_rotation_normalized = joint_axis_normalized - joint_translation_normalized
            # joint_rotation_normalized /= np.linalg.norm(joint_rotation_normalized)
            joint_rotation_normalized = joint_rotations[j].copy()
            joint_rotation_normalized /= np.linalg.norm(joint_rotation_normalized)
            joint_rotations_normalized.append(joint_rotation_normalized)
    if joint_translations is not None:
        joint_translations_normalized = np.array(joint_translations_normalized).astype(joint_translations.dtype)
    else:
        joint_translations_normalized = None
    if joint_rotations is not None:
        joint_rotations_normalized = np.array(joint_rotations_normalized).astype(joint_rotations.dtype)
    else:
        joint_rotations_normalized = None
    return (joint_translations_normalized, joint_rotations_normalized)

def joints_denormalize(joint_translations_normalized:Optional[np.ndarray], joint_rotations_normalized:Optional[np.ndarray], center:np.ndarray, scale:float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if joint_translations_normalized is None and joint_rotations_normalized is None:
        return (None, None)
    
    J = joint_translations_normalized.shape[0] if joint_translations_normalized is not None else joint_rotations_normalized.shape[0]
    joint_translations, joint_rotations = [], []
    for j in range(J):
        if joint_translations_normalized is not None:
            joint_translation = joint_translations_normalized[j] * scale + center
            joint_translations.append(joint_translation)
        if joint_rotations_normalized is not None:
            # joint_axis = (joint_translations_normalized[j] + joint_rotations_normalized[j]) * scale + center
            # joint_rotation = joint_axis - joint_translation
            joint_rotation = joint_rotations_normalized[j].copy()
            joint_rotation /= np.linalg.norm(joint_rotation)
            joint_rotations.append(joint_rotation)
    if joint_translations_normalized is not None:
        joint_translations = np.array(joint_translations).astype(joint_translations_normalized.dtype)
    else:
        joint_translations = None
    if joint_rotations_normalized is not None:
        joint_rotations = np.array(joint_rotations).astype(joint_rotations_normalized.dtype)
    else:
        joint_rotations = None
    return (joint_translations, joint_rotations)

def joint_denormalize(joint_translation_normalized:Optional[np.ndarray], joint_rotation_normalized:Optional[np.ndarray], center:np.ndarray, scale:float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if joint_translation_normalized is not None:
        joint_translation = joint_translation_normalized * scale + center
    else:
        joint_translation = None
    if joint_rotation_normalized is not None:
        # joint_axis = (joint_translation_normalized + joint_rotation_normalized) * scale + center
        # joint_rotation = joint_axis - joint_translation
        joint_rotation = joint_rotation_normalized.copy()
        joint_rotation /= np.linalg.norm(joint_rotation)
    else:
        joint_rotation = None
    return (joint_translation, joint_rotation)

def transform_pc(pc_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # pc_camera: (N, 3), c2w: (4, 4)
    pc_camera_hm = np.concatenate([pc_camera, np.ones((pc_camera.shape[0], 1), dtype=pc_camera.dtype)], axis=-1)        # (N, 4)
    pc_world_hm = pc_camera_hm @ c2w.T                                                          # (N, 4)
    pc_world = pc_world_hm[:, :3]                                                               # (N, 3)
    return pc_world

def transform_dir(dir_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # dir_camera: (N, 3), c2w: (4, 4)
    dir_camera_hm = np.concatenate([dir_camera, np.zeros((dir_camera.shape[0], 1), dtype=dir_camera.dtype)], axis=-1)   # (N, 4)
    dir_world_hm = dir_camera_hm @ c2w.T                                                        # (N, 4)
    dir_world = dir_world_hm[:, :3]                                                             # (N, 3)
    return dir_world


def generate_target_tr(pc:np.ndarray, o:np.ndarray, point_idxs:np.ndarray) -> np.ndarray:
    a = pc[point_idxs[:, 0]]    # (N_t, 3)
    b = pc[point_idxs[:, 1]]    # (N_t, 3)
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    proj_len = np.sum((a - o) * pdist_unit, -1)
    oc = a - o - proj_len[..., None] * pdist_unit
    dist2o = np.linalg.norm(oc, axis=-1)
    target_tr = np.stack([proj_len, dist2o], -1)
    return target_tr.astype(np.float32).reshape((-1, 2))
    
def generate_target_rot(pc:np.ndarray, axis:np.ndarray, point_idxs:np.ndarray) -> np.ndarray:
    a = pc[point_idxs[:, 0]]    # (N_t, 3)
    b = pc[point_idxs[:, 1]]    # (N_t, 3)
    pdist = a - b
    pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
    cos = np.sum(pdist_unit * axis, axis=-1)
    cos = np.clip(cos, -1., 1.)
    target_rot = np.arccos(cos)
    return target_rot.astype(np.float32).reshape((-1,))


def farthest_point_sample(point:np.ndarray, npoint:int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        point: sampled pointcloud, [npoint, D]
        centroids: sampled pointcloud index
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    return (point, centroids)


def fibonacci_sphere(samples:int) -> List[Tuple[float, float, float]]:
    points = []
    phi = math.pi * (3. - math.sqrt(5.))    # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)   # radius at y

        theta = phi * i # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def real2prob(val:Union[torch.Tensor, np.ndarray], max_val:float, num_bins:int, circular:bool=False) -> Union[torch.Tensor, np.ndarray]:
    is_torch = isinstance(val, torch.Tensor)
    if is_torch:
        res = torch.zeros((*val.shape, num_bins), dtype=val.dtype).to(val.device)
    else:
        res = np.zeros((*val.shape, num_bins), dtype=val.dtype)
        
    if not circular:
        interval = max_val / (num_bins - 1)
        if is_torch:
            low = torch.clamp(torch.floor(val / interval).long(), max=num_bins - 2)
        else:
            low = np.clip(np.floor(val / interval).astype(np.int64), a_min=None, a_max=num_bins - 2)
        high = low + 1
        # assert torch.all(low >= 0) and torch.all(high < num_bins)
        
        # huge memory
        if is_torch:
            res.scatter_(-1, low[..., None], torch.unsqueeze(1. - (val / interval - low), -1))
            res.scatter_(-1, high[..., None], 1. - torch.gather(res, -1, low[..., None]))
        else:
            np.put_along_axis(res, low[..., None], np.expand_dims(1. - (val / interval - low), -1), -1)
            np.put_along_axis(res, high[..., None], 1. - np.take_along_axis(res, low[..., None], -1), -1)
        # res[..., low] = 1. - (val / interval - low)
        # res[..., high] = 1. - res[..., low]
        # assert torch.all(0 <= res[..., low]) and torch.all(1 >= res[..., low])
        return res
    else:
        interval = max_val / num_bins
        if is_torch:
            val_new = torch.clone(val)
        else:
            val_new = val.copy()
        val_new[val < interval / 2] += max_val
        res = real2prob(val_new - interval / 2, max_val, num_bins + 1)
        res[..., 0] += res[..., -1]
        return res[..., :-1]


def pc_ncs(pc:np.ndarray, bbox_min:np.ndarray, bbox_max:np.ndarray) -> np.ndarray:
    return (pc - (bbox_min + bbox_max) / 2 + 0.5 * (bbox_max - bbox_min)) / (bbox_max - bbox_min)

def joints_ncs(joint_translations:np.ndarray, joint_rotations:Optional[np.ndarray], bbox_min:np.ndarray, bbox_max:np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return ((joint_translations - (bbox_min + bbox_max) / 2 + 0.5 * (bbox_max - bbox_min)) / (bbox_max - bbox_min), joint_rotations.copy() if joint_rotations is not None else None)


def rotate_points_with_rotvec(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def scale_pts(source, target):
    # compute scaling factor between source: [N x 3], target: [N x 3]
    pdist_s = source.reshape(source.shape[0], 1, 3) - source.reshape(1, source.shape[0], 3)
    A = np.sqrt(np.sum(pdist_s**2, 2)).reshape(-1)
    pdist_t = target.reshape(target.shape[0], 1, 3) - target.reshape(1, target.shape[0], 3)
    b = np.sqrt(np.sum(pdist_t**2, 2)).reshape(-1)
    scale = np.dot(A, b) / (np.dot(A, A)+1e-6)
    return scale

def rotate_pts(source, target):
    # compute rotation between source: [N x 3], target: [N x 3]
    # pre-centering
    source = source - np.mean(source, 0, keepdims=True)
    target = target - np.mean(target, 0, keepdims=True)
    M = np.matmul(target.T, source)
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = np.matmul(U, Vh)
    return R


def objective_eval_t(params, x0, y0, x1, y1, joints, isweight=True):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1,3))
    rotvec1 = params[3:].reshape((1,3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_R = rotvec0 - rotvec1
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
    return np.concatenate((res0, res1, res_R), 0).ravel()

def objective_eval_r(params, x0, y0, x1, y1, joints, isweight=True):
    # params: [:3] R0, [3:] R1
    # x0: N x 3, y0: N x 3, x1: M x 3, y1: M x 3, R0: 1 x 3, R1: 1 x 3, joints: K x 3
    rotvec0 = params[:3].reshape((1,3))
    rotvec1 = params[3:].reshape((1,3))
    res0 = y0 - rotate_points_with_rotvec(x0, rotvec0)
    res1 = y1 - rotate_points_with_rotvec(x1, rotvec1)
    res_joint = rotate_points_with_rotvec(joints, rotvec0) - rotate_points_with_rotvec(joints, rotvec1)
    if isweight:
        res0 /= x0.shape[0]
        res1 /= x1.shape[0]
        res_joint /= joints.shape[0]
    return np.concatenate((res0, res1, res_joint), 0).ravel()


def joint_transformation_estimator(dataset:Dict[str, np.ndarray], joint_type:str, best_inliers:Optional[Tuple[np.ndarray, np.ndarray]]=None) -> Optional[Dict[str, np.ndarray]]:
    nsource0 = dataset['source0'].shape[0]
    nsource1 = dataset['source1'].shape[0]
    if nsource0 < 3 or nsource1 < 3:
        return None
    if best_inliers is None:
        sample_idx0 = np.random.randint(nsource0, size=3)
        sample_idx1 = np.random.randint(nsource1, size=3)
    else:
        sample_idx0 = best_inliers[0]
        sample_idx1 = best_inliers[1]

    source0 = dataset['source0'][sample_idx0, :]
    target0 = dataset['target0'][sample_idx0, :]
    source1 = dataset['source1'][sample_idx1, :]
    target1 = dataset['target1'][sample_idx1, :]

    scale0 = scale_pts(source0, target0)
    scale1 = scale_pts(source1, target1)
    scale0_inv = scale_pts(target0, source0)
    scale1_inv = scale_pts(target1, source1)

    target0_scaled_centered = scale0_inv*target0
    target0_scaled_centered -= np.mean(target0_scaled_centered, 0, keepdims=True)
    source0_centered = source0 - np.mean(source0, 0, keepdims=True)

    target1_scaled_centered = scale1_inv*target1
    target1_scaled_centered -= np.mean(target1_scaled_centered, 0, keepdims=True)
    source1_centered = source1 - np.mean(source1, 0, keepdims=True)

    joint_points0 = np.ones_like(np.linspace(0, 1, num = np.min((source0.shape[0], source1.shape[0]))+1 )[1:].reshape((-1, 1)))*dataset['joint_direction'].reshape((1, 3))

    R0 = rotate_pts(source0_centered, target0_scaled_centered)
    R1 = rotate_pts(source1_centered, target1_scaled_centered)

    rotvec0 = srot.from_matrix(R0).as_rotvec()
    rotvec1 = srot.from_matrix(R1).as_rotvec()
    if joint_type == 'prismatic':
        res = least_squares(objective_eval_t, np.hstack((rotvec0, rotvec1)), verbose=0, ftol=1e-4, method='lm',
                        args=(source0_centered, target0_scaled_centered, source1_centered, target1_scaled_centered, joint_points0, False))
    elif joint_type == 'revolute':
        res = least_squares(objective_eval_r, np.hstack((rotvec0, rotvec1)), verbose=0, ftol=1e-4, method='lm',
                        args=(source0_centered, target0_scaled_centered, source1_centered, target1_scaled_centered, joint_points0, False))
    else:
        raise ValueError
    R0 = srot.from_rotvec(res.x[:3]).as_matrix()
    R1 = srot.from_rotvec(res.x[3:]).as_matrix()

    translation0 = np.mean(target0.T-scale0*np.matmul(R0, source0.T), 1)
    translation1 = np.mean(target1.T-scale1*np.matmul(R1, source1.T), 1)

    if np.isnan(translation0).any() or np.isnan(translation1).any() or np.isnan(R0).any() or np.isnan(R0).any():
        return None

    jtrans = dict()
    jtrans['rotation0'] = R0
    jtrans['scale0'] = scale0
    jtrans['translation0'] = translation0
    jtrans['rotation1'] = R1
    jtrans['scale1'] = scale1
    jtrans['translation1'] = translation1
    return jtrans

def joint_transformation_verifier(dataset:Dict[str, np.ndarray], model:Dict[str, np.ndarray], inlier_th:float) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    res0 = dataset['target0'].T - model['scale0'] * np.matmul( model['rotation0'], dataset['source0'].T ) - model['translation0'].reshape((3, 1))
    inliers0 = np.sqrt(np.sum(res0**2, 0)) < inlier_th
    res1 = dataset['target1'].T - model['scale1'] * np.matmul( model['rotation1'], dataset['source1'].T ) - model['translation1'].reshape((3, 1))
    inliers1 = np.sqrt(np.sum(res1**2, 0)) < inlier_th
    score = ( np.sum(inliers0)/res0.shape[0] + np.sum(inliers1)/res1.shape[0] ) / 2
    return (score, (inliers0, inliers1))

def ransac(dataset:Dict[str, np.ndarray], inlier_threshold:float, iteration_num:int, joint_type:str) -> Optional[Dict[str, np.ndarray]]:
    best_model = None
    best_score = -np.inf
    best_inliers = None
    for i in range(iteration_num):
        cur_model = joint_transformation_estimator(dataset, joint_type=joint_type, best_inliers=None)
        if cur_model is None:
            return None
        cur_score, cur_inliers = joint_transformation_verifier(dataset, cur_model, inlier_threshold)
        if cur_score > best_score:
            best_model = cur_model
            best_inliers = cur_inliers
            best_score = cur_score
    best_model = joint_transformation_estimator(dataset, joint_type=joint_type, best_inliers=best_inliers)
    return best_model


def match_joints(proposal_joints:List[Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]], 
                 gt_joint_translations:np.ndarray, gt_joint_rotations:np.ndarray, match_metric:str, 
                 has_affordance:bool=False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # TODO: affordance-related not very well
    proposal_num = len(proposal_joints)                 # N
    gt_num = gt_joint_translations.shape[0]             # M
    if proposal_num == 0:
        pred_joint_translations = np.zeros_like(gt_joint_translations, dtype=gt_joint_translations.dtype)
        pred_joint_rotations = np.ones_like(gt_joint_rotations, dtype=gt_joint_rotations.dtype)
        pred_joint_rotations = pred_joint_rotations / np.linalg.norm(pred_joint_rotations, axis=-1, keepdims=True)
        if has_affordance:
            pred_affordances = np.zeros_like(gt_joint_translations, dtype=gt_joint_translations.dtype)
            return (pred_joint_translations, pred_joint_rotations, pred_affordances)
        else:
            return (pred_joint_translations, pred_joint_rotations)

    proposal_joint_translations = np.array([proposal_joints[i][0] for i in range(proposal_num)])     # (N, 3)
    proposal_joint_rotations = np.array([proposal_joints[i][1] for i in range(proposal_num)])        # (N, 3)
    if has_affordance:
        proposal_affordances = np.array([proposal_joints[i][2] for i in range(proposal_num)])        # (N, 3)
    else:
        pass
    cost_matrix = np.zeros((gt_num, proposal_num))      # (M, N)
    for gt_idx in range(gt_num):
        gt_joint_translation = gt_joint_translations[gt_idx].reshape((1, 3)).repeat(proposal_num, axis=0)   # (N, 3)
        gt_joint_rotation = gt_joint_rotations[gt_idx].reshape((1, 3)).repeat(proposal_num, axis=0)         # (N, 3)
        translation_errors = calc_translation_error_batch(proposal_joint_translations, gt_joint_translation, proposal_joint_rotations, gt_joint_rotation)   # (N,)
        direction_errors = calc_direction_error_batch(proposal_joint_rotations, gt_joint_rotation)          # (N,)
        if match_metric == 'tr_dist':
            cost_matrix[gt_idx] = translation_errors[0]
        elif match_metric == 'tr_along':
            cost_matrix[gt_idx] = translation_errors[1]
        elif match_metric == 'tr_perp':
            cost_matrix[gt_idx] = translation_errors[2]
        elif match_metric == 'tr_plane':
            cost_matrix[gt_idx] = translation_errors[3]
        elif match_metric == 'tr_line':
            cost_matrix[gt_idx] = translation_errors[4]
        elif match_metric == 'tr_mean':
            cost_matrix[gt_idx] = (translation_errors[0] + translation_errors[1] + translation_errors[2] + translation_errors[3] + translation_errors[4]) / 5
        elif match_metric == 'dir':
            cost_matrix[gt_idx] = direction_errors
        elif match_metric == 'tr_dist_dir':
            cost_matrix[gt_idx] = (translation_errors[0] + direction_errors) / 2
        elif match_metric == 'tr_line_dir':
            cost_matrix[gt_idx] = (translation_errors[4] + direction_errors) / 2
        elif match_metric == 'tr_mean_dir':
            cost_matrix[gt_idx] = ((translation_errors[0] + translation_errors[1] + translation_errors[2] + translation_errors[3] + translation_errors[4]) / 5 + direction_errors) / 2
        else:
            raise NotImplementedError
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pred_joint_translations = np.zeros_like(gt_joint_translations, dtype=gt_joint_translations.dtype)
    pred_joint_rotations = np.ones_like(gt_joint_rotations, dtype=gt_joint_rotations.dtype)
    if has_affordance:
        pred_affordances = np.zeros_like(gt_joint_translations, dtype=gt_joint_translations.dtype)
    else:
        pass
    for gt_idx, proposal_idx in zip(row_ind, col_ind):
        pred_joint_translations[gt_idx] = proposal_joint_translations[proposal_idx]
        pred_joint_rotations[gt_idx] = proposal_joint_rotations[proposal_idx]
        if has_affordance:
            pred_affordances[gt_idx] = proposal_affordances[proposal_idx]
        else:
            pass
    pred_joint_rotations = pred_joint_rotations / np.linalg.norm(pred_joint_rotations, axis=-1, keepdims=True)
    if has_affordance:
        return (pred_joint_translations, pred_joint_rotations, pred_affordances)
    else:
        return (pred_joint_translations, pred_joint_rotations)


def pc_noise(pc:np.ndarray, distortion_rate:float, distortion_level:float, 
             outlier_rate:float, outlier_level:float) -> np.ndarray:
    num_points = pc.shape[0]
    pc_center = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2
    pc_scale = np.max(pc, axis=0) - np.min(pc, axis=0)
    pc_noised = pc.copy()
    # distortion noise
    distortion_indices = np.random.choice(num_points, int(distortion_rate * num_points), replace=False)
    distortion_noise = np.random.normal(0.0, distortion_level * pc_scale, pc_noised[distortion_indices].shape)
    pc_noised[distortion_indices] = pc_noised[distortion_indices] + distortion_noise
    # outlier noise
    outlier_indices = np.random.choice(num_points, int(outlier_rate * num_points), replace=False)
    outlier_noise = np.random.uniform(pc_center - outlier_level * pc_scale, pc_center + outlier_level * pc_scale, pc_noised[outlier_indices].shape)
    pc_noised[outlier_indices] = outlier_noise
    # print(num_points, int(distortion_rate * num_points), int(outlier_rate * num_points))
    return pc_noised


if __name__ == '__main__':
    # angle_tol = 1.5
    angle_tol = 0.35
    rot_candidate_num = int(4 * np.pi / (angle_tol / 180 * np.pi))
    print(rot_candidate_num)
    sphere_pts = fibonacci_sphere(rot_candidate_num)    # (N, 3)
    sphere_pts = np.array(sphere_pts)

    # figure out the angles between neighboring 8 points on the sphere
    min_angles = []
    max_angles = []
    mean_angles = []
    for i in range(len(sphere_pts)):
        a = sphere_pts[i]                               # (3,)
        cos = np.dot(a, sphere_pts.T)                   # (N,)
        cos = np.clip(cos, -1., 1.)
        theta = np.arccos(cos)                          # (N,)
        idxs = np.argsort(theta)[:9]                    # (9,)
        thetas = theta[idxs[1:]] / np.pi * 180          # (8,)
        min_angles.append(np.min(thetas))
        max_angles.append(np.max(thetas))
        mean_angles.append(np.mean(thetas))

    print(np.min(min_angles))
    print(np.max(min_angles))
    print(np.mean(min_angles))
    print(np.min(max_angles))
    print(np.max(max_angles))
    print(np.mean(max_angles))
    print(np.min(mean_angles))
    print(np.max(mean_angles))
    print(np.mean(mean_angles))
