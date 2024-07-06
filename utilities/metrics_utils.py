from typing import Tuple, List, Union, Dict, Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
from logging import Logger
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def calc_translation_error(pred_p:np.ndarray, gt_p:np.ndarray, pred_e:Optional[np.ndarray], gt_e:Optional[np.ndarray]) -> Tuple[float, float, float, float, float]:
    def calc_plane_error(pred_translation:np.ndarray, pred_direction:np.ndarray, 
                         gt_translation:np.ndarray, gt_direction:np.ndarray) -> float:
        if abs(np.dot(pred_direction, gt_direction)) < 1e-3:
            # parallel to the plane
            # point-to-line distance
            dist = np.linalg.norm(np.cross(pred_direction, gt_translation - pred_translation))
            return dist
        # gt_direction \dot (x - gt_translation) = 0
        # x = pred_translation + t * pred_direction
        t = np.dot(gt_translation - pred_translation, gt_direction) / np.dot(pred_direction, gt_direction)
        x = pred_translation + t * pred_direction
        dist = np.linalg.norm(x - gt_translation)
        return dist
    def calc_line_error(pred_translation:np.ndarray, pred_direction:np.ndarray, 
                        gt_translation:np.ndarray, gt_direction:np.ndarray) -> float:
        orth_vect = np.cross(gt_direction, pred_direction)
        p = gt_translation - pred_translation
        if np.linalg.norm(orth_vect) < 1e-3:
            dist = np.linalg.norm(np.cross(p, gt_direction)) / np.linalg.norm(gt_direction)
        else:
            dist = abs(np.dot(orth_vect, p)) / np.linalg.norm(orth_vect)
        return dist
    distance_error = np.linalg.norm(pred_p - gt_p) * 100.0
    if pred_e is None or gt_e is None:
        along_error = 0.0
        perp_error = 0.0
        plane_error = 0.0
        line_error = 0.0
    else:
        along_error = abs(np.dot(pred_p - gt_p, gt_e)) * 100.0
        perp_error = np.sqrt(distance_error**2 - along_error**2)
        plane_error = calc_plane_error(pred_p, pred_e, gt_p, gt_e) * 100.0
        line_error = calc_line_error(pred_p, pred_e, gt_p, gt_e) * 100.0

    return (distance_error, along_error, perp_error, plane_error, line_error)

def calc_translation_error_batch(pred_ps:np.ndarray, gt_ps:np.ndarray, pred_es:Optional[np.ndarray], gt_es:Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def calc_plane_error_batch(pred_translations:np.ndarray, pred_directions:np.ndarray, 
                               gt_translations:np.ndarray, gt_directions:np.ndarray) -> np.ndarray:
        dists = np.zeros(pred_translations.shape[:-1], dtype=pred_translations.dtype)
        flag = np.abs(np.sum(pred_directions * gt_directions, axis=-1)) < 1e-3
        not_flag = np.logical_not(flag)

        dists[flag] = np.linalg.norm(np.cross(gt_translations[flag] - pred_translations[flag], pred_translations[flag]), axis=-1)

        ts = np.sum((gt_translations[not_flag] - pred_translations[not_flag]) * gt_directions[not_flag], axis=-1) / np.sum(pred_directions[not_flag] * gt_directions[not_flag], axis=-1)
        xs = pred_translations[not_flag] + ts[..., None] * pred_directions[not_flag]
        dists[not_flag] = np.linalg.norm(xs - gt_translations[not_flag], axis=-1)
        return dists
    def calc_line_error_batch(pred_translations:np.ndarray, pred_directions:np.ndarray, 
                              gt_translations:np.ndarray, gt_directions:np.ndarray) -> np.ndarray:
        dists = np.zeros(pred_translations.shape[:-1], dtype=pred_translations.dtype)
        orth_vects = np.cross(gt_directions, pred_directions)
        ps = gt_translations - pred_translations
        flag = np.linalg.norm(orth_vects, axis=-1) < 1e-3
        not_flag = np.logical_not(flag)

        dists[flag] = np.linalg.norm(np.cross(ps[flag], gt_directions[flag]), axis=-1) / np.linalg.norm(gt_directions[flag], axis=-1)

        dists[not_flag] = np.abs(np.sum(orth_vects[not_flag] * ps[not_flag], axis=-1)) / np.linalg.norm(orth_vects[not_flag], axis=-1)
        return dists
    distance_errors = np.linalg.norm(pred_ps - gt_ps, axis=-1) * 100.0
    if pred_es is None or gt_es is None:
        along_errors = np.zeros(distance_errors.shape, dtype=distance_errors.dtype)
        perp_errors = np.zeros(distance_errors.shape, dtype=distance_errors.dtype)
        plane_errors = np.zeros(distance_errors.shape, dtype=distance_errors.dtype)
        line_errors = np.zeros(distance_errors.shape, dtype=distance_errors.dtype)
    else:
        along_errors = np.abs(np.sum((pred_ps - gt_ps) * gt_es, axis=-1)) * 100.0
        perp_errors = np.sqrt(distance_errors**2 - along_errors**2)
        plane_errors = calc_plane_error_batch(pred_ps, pred_es, gt_ps, gt_es) * 100.0
        line_errors = calc_line_error_batch(pred_ps, pred_es, gt_ps, gt_es) * 100.0

    return (distance_errors, along_errors, perp_errors, plane_errors, line_errors)

def calc_direction_error(pred_e:np.ndarray, gt_e:np.ndarray) -> float:
    cos_theta = np.dot(pred_e, gt_e)
    cos_theta = np.clip(cos_theta, -1., 1.)
    angle_radian = np.arccos(cos_theta) * 180.0 / np.pi
    return angle_radian

def calc_direction_error_batch(pred_es:np.ndarray, gt_es:np.ndarray) -> np.ndarray:
    cos_thetas = np.sum(pred_es * gt_es, axis=-1)
    cos_thetas = np.clip(cos_thetas, -1., 1.)
    angle_radians = np.arccos(cos_thetas) * 180.0 / np.pi
    return angle_radians


def calculate_miou_loss(pred_seg:torch.Tensor, gt_seg_onehot:torch.Tensor) -> torch.Tensor:
    # pred_seg: (B, N, C), gt_seg_onehot: (B, N, C)
    dot = torch.sum(pred_seg * gt_seg_onehot, dim=-2)
    denominator = torch.sum(pred_seg, dim=-2) + torch.sum(gt_seg_onehot, dim=-2) - dot
    mIoU = dot / (denominator + 1e-7)                                               # (B, C)
    return torch.mean(1.0 - mIoU)

def calculate_ncs_loss(pred_ncs:torch.Tensor, gt_ncs:torch.Tensor, gt_seg_onehot:torch.Tensor) -> torch.Tensor:
    # pred_ncs: (B, N, 3 * C), gt_ncs: (B, N, 3), gt_seg_onehot: (B, N, C)
    loss_ncs = 0
    ncs_splits = torch.split(pred_ncs, split_size_or_sections=3, dim=-1)        # (C,), (B, N, 3)
    seg_splits = torch.split(gt_seg_onehot, split_size_or_sections=1, dim=2)    # (C,), (B, N, 1)
    C = len(ncs_splits)
    for i in range(C):
        diff_l2 = torch.norm(ncs_splits[i] - gt_ncs, dim=-1)                    # (B, N)
        loss_ncs += torch.mean(seg_splits[i][:, :, 0] * diff_l2, dim=-1)        # (B,)
    return torch.mean(loss_ncs)

def calculate_heatmap_loss(pred_heatmap:torch.Tensor, gt_heatmap:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    # pred_heatmap: (B, N), gt_heatmap: (B, N), mask: (B, N)
    loss_heatmap = torch.abs(pred_heatmap - gt_heatmap)[mask > 0]
    loss_heatmap = torch.mean(loss_heatmap)
    loss_heatmap[torch.isnan(loss_heatmap)] = 0.0
    return loss_heatmap

def calculate_unitvec_loss(pred_unitvec:torch.Tensor, gt_unitvec:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    # pred_unitvec: (B, N, 3), gt_unitvec: (B, N, 3), mask: (B, N)
    loss_unitvec = torch.norm(pred_unitvec - gt_unitvec, dim=-1)[mask > 0]
    return torch.mean(loss_unitvec)
    # pt_diff = pred_unitvec - gt_unitvec
    # pt_dist = torch.sum(pt_diff.abs(), dim=-1)          # (B, N)
    # loss_pt_dist = torch.mean(pt_dist[mask > 0])
    # pred_unitvec_normalized = pred_unitvec / (torch.norm(pred_unitvec, dim=-1, keepdim=True) + 1e-7)
    # gt_unitvec_normalized = gt_unitvec / (torch.norm(gt_unitvec, dim=-1, keepdim=True) + 1e-7)
    # dir_diff = torch.sum(-(pred_unitvec_normalized * gt_unitvec_normalized), dim=-1)    # (B, N)
    # loss_dir_diff = torch.mean(dir_diff[mask > 0])
    # loss_unitvec = loss_pt_dist + loss_dir_diff
    # return loss_unitvec


def focal_loss(inputs:torch.Tensor, targets:torch.Tensor, alpha:Optional[torch.Tensor]=None, gamma:float=2.0, ignore_index:Optional[int]=None) -> torch.Tensor:
    # inputs: (N, C), targets: (N,)
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        targets = targets[valid_mask]

        if targets.shape[0] == 0:
            return torch.tensor(0.0).to(dtype=inputs.dtype, device=inputs.device)

        inputs = inputs[valid_mask]

    log_p = torch.clamp(torch.log(inputs + 1e-7), max=0.0)
    if ignore_index is not None:
        ce_loss = F.nll_loss(
            log_p, targets, weight=alpha, ignore_index=ignore_index, reduction="none"
        )
    else:
        ce_loss = F.nll_loss(
            log_p, targets, weight=alpha, reduction="none"
        )
    log_p_t = log_p.gather(1, targets[:, None]).squeeze(-1)
    loss = ce_loss * ((1 - log_p_t.exp()) ** gamma)
    loss = loss.mean()
    return loss

def dice_loss(input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
    def one_hot(labels:torch.Tensor, num_classes:int, device:Optional[torch.device]=None, dtype:Optional[torch.dtype]=None) -> torch.Tensor:
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

        if not labels.dtype == torch.int64:
            raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

        if num_classes < 1:
            raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

        shape = labels.shape
        one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + 1e-6
    # input: (N, C, H, W), target: (N, H, W)
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # create the labels one hot tensor
    target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input * target_one_hot, dims)
    cardinality = torch.sum(input + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + 1e-8)

    return torch.mean(-dice_score + 1.0)


def calculate_offset_loss(pred_offsets:torch.Tensor, gt_offsets:torch.Tensor, function_masks:torch.Tensor) -> torch.Tensor:
    # pred_offsets: (B, N, 3), gt_offsets: (B, N, 3), function_masks: (B, N)
    pt_diff = pred_offsets - gt_offsets
    pt_dist = torch.sum(pt_diff.abs(), dim=-1)                                  # (B, N)
    loss_pt_offset_dist = pt_dist[function_masks < 2].mean()
    loss_pt_offset_dist[torch.isnan(loss_pt_offset_dist)] = 0.0
    gt_offsets_norm = torch.norm(gt_offsets, dim=-1, keepdim=True)
    gt_offsets_normalized = gt_offsets / (gt_offsets_norm + 1e-8)
    pred_offsets_norm = torch.norm(pred_offsets, dim=-1, keepdim=True)
    pred_offsets_normalized = pred_offsets / (pred_offsets_norm + 1e-8)
    dir_diff = -(gt_offsets_normalized * pred_offsets_normalized).sum(dim=-1)   # (B, N)
    loss_offset_dir = dir_diff[function_masks < 2].mean()
    loss_offset_dir[torch.isnan(loss_offset_dir)] = 0.0
    loss_offset = loss_offset_dir + loss_pt_offset_dist
    return loss_offset

def calculate_dir_loss(pred_dirs:torch.Tensor, gt_dirs:torch.Tensor, function_masks:torch.Tensor) -> torch.Tensor:
    # pred_dirs: (B, N, 3), gt_dirs: (B, N, 3), function_masks: (B, N)
    pt_diff = pred_dirs - gt_dirs
    pt_dist = torch.sum(pt_diff.abs(), dim=-1)      # (B, N)
    dis_loss = pt_dist[function_masks < 2].mean()
    dis_loss[torch.isnan(dis_loss)] = 0.0
    dir_loss = -(pred_dirs * gt_dirs).sum(dim=-1)   # (B, N)
    dir_loss = dir_loss[function_masks < 2].mean()
    dir_loss[torch.isnan(dir_loss)] = 0.0
    loss = dis_loss + dir_loss
    return loss


def calc_pose_error(pose1:np.ndarray, pose2:np.ndarray) -> Tuple[float, float]:
    error_matrix = np.dot(np.linalg.inv(pose1), pose2)
    translation_error = error_matrix[:3, 3]
    rotation_error = np.arccos(np.clip((np.trace(error_matrix[:3, :3]) - 1) / 2, -1, 1))
    return (np.linalg.norm(translation_error), rotation_error)


def invaffordance_metrics(grasp_translation:np.ndarray, grasp_rotation:np.ndarray, grasp_score:float,  affordable_position:np.ndarray, 
                          joint_base:np.ndarray, joint_direction:np.ndarray, joint_type:int) -> Tuple[float, float, float]:
    if joint_type == 0:
        # revolute
        l2_dist = np.linalg.norm(grasp_translation - affordable_position)
        # normal = np.cross(joint_base - affordable_position, joint_base + joint_direction - affordable_position)
        # normal = normal / np.linalg.norm(normal)
        # plane_dist = abs(np.dot(grasp_translation - affordable_position, normal))
        # return (1.0 - grasp_score,)
        # return (l2_dist, plane_dist, 1.0 - grasp_score)
        return (l2_dist,)
    elif joint_type == 1:
        # prismatic
        l2_dist = np.linalg.norm(grasp_translation - affordable_position)
        # plane_dist = abs(np.dot(grasp_translation - affordable_position, joint_direction))
        # return (1.0 - grasp_score,)
        # return (l2_dist, plane_dist, 1.0 - grasp_score)
        return (l2_dist,)
    else:
        raise ValueError(f"Invalid joint_type: {joint_type}")

def invaffordances2affordance(invaffordances:List[Tuple[float, float, float]], sort:bool=False) -> List[float]:
    if len(invaffordances) == 0:
        return []
    if len(invaffordances) == 1:
        return [1.0]
    # from list of tuple to dict of list, k: metrics index, v: invaffordance list
    invaffordances_dict = {k: [v[k] for v in invaffordances] for k in range(len(invaffordances[0]))}
    if sort:
        # sort each list in dict
        affordances_dict = {}
        for k in invaffordances_dict.keys():
            affordances_dict[k] = len(invaffordances_dict[k]) - 1 - np.argsort(np.argsort(invaffordances_dict[k]))
            affordances_dict[k] /= len(invaffordances) - 1
    else:
        # normalize each list in dict
        affordances_dict = {}
        for k in invaffordances_dict.keys():
            affordances_dict[k] = (np.max(invaffordances_dict[k]) - invaffordances_dict[k]) / (np.max(invaffordances_dict[k]) - np.min(invaffordances_dict[k]))
    # merge into list
    affordances = np.array([0.0 for _ in range(len(invaffordances))])
    for k in affordances_dict.keys():
        affordances += affordances_dict[k]
    affordances /= len(affordances_dict.keys())
    return affordances.tolist()


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_metrics(results_dict:Dict[str, Union[np.ndarray, int, List[str]]], logger:Logger, log_path:str, 
                tb_writer:Optional[SummaryWriter]=None, epoch:Optional[int]=None, split:Optional[str]=None) -> None:
    translation_distance_errors = results_dict['translation_distance_errors']
    translation_along_errors = results_dict['translation_along_errors']
    translation_perp_errors = results_dict['translation_perp_errors']
    translation_plane_errors = results_dict['translation_plane_errors']
    translation_line_errors = results_dict['translation_line_errors']
    translation_outliers_num = results_dict['translation_outliers_num']
    rotation_errors = results_dict['rotation_errors']
    rotation_outliers_num = results_dict['rotation_outliers_num']
    if 'affordance_errors' in results_dict.keys():
        affordance_errors = results_dict['affordance_errors']
        affordance_outliers_num = results_dict['affordance_outliers_num']
    else:
        affordance_errors = [0, 0]
        affordance_outliers_num = 0

    if len(translation_distance_errors.shape) == 1:
        data_num = translation_distance_errors.shape[0]
        if 'names' in results_dict.keys():
            names = results_dict['names']
            dataframe = pd.DataFrame({'name': names, 
                                      'translation_distance_errors': translation_distance_errors, 
                                      'translation_along_errors': translation_along_errors, 
                                      'translation_perp_errors': translation_perp_errors, 
                                      'translation_plane_errors': translation_plane_errors, 
                                      'translation_line_errors': translation_line_errors, 
                                      'rotation_errors': rotation_errors, 
                                      'affordance_errors': affordance_errors})
            dataframe.to_csv(os.path.join(log_path, 'metrics.csv'), sep=',')
        else:
            pass

        # mean
        mean_translation_distance_error = np.mean(translation_distance_errors, axis=0)
        mean_translation_along_error = np.mean(translation_along_errors, axis=0)
        mean_translation_perp_error = np.mean(translation_perp_errors, axis=0)
        mean_translation_plane_error = np.mean(translation_plane_errors, axis=0)
        mean_translation_line_error = np.mean(translation_line_errors, axis=0)
        mean_rotation_error = np.mean(rotation_errors, axis=0)
        mean_affordance_error = np.mean(affordance_errors, axis=0)
        logger.info(f"mean_translation_distance_error = {mean_translation_distance_error}")
        logger.info(f"mean_translation_along_error = {mean_translation_along_error}")
        logger.info(f"mean_translation_perp_error = {mean_translation_perp_error}")
        logger.info(f"mean_translation_plane_error = {mean_translation_plane_error}")
        logger.info(f"mean_translation_line_error = {mean_translation_line_error}")
        logger.info(f"mean_rotation_error = {mean_rotation_error}")
        logger.info(f"mean_affordance_error = {mean_affordance_error}")
        # median
        median_translation_distance_error = np.median(translation_distance_errors, axis=0)
        median_translation_along_error = np.median(translation_along_errors, axis=0)
        median_translation_perp_error = np.median(translation_perp_errors, axis=0)
        median_translation_plane_error = np.median(translation_plane_errors, axis=0)
        median_translation_line_error = np.median(translation_line_errors, axis=0)
        median_rotation_error = np.median(rotation_errors, axis=0)
        median_affordance_error = np.median(affordance_errors, axis=0)
        logger.info(f"median_translation_distance_error = {median_translation_distance_error}")
        logger.info(f"median_translation_along_error = {median_translation_along_error}")
        logger.info(f"median_translation_perp_error = {median_translation_perp_error}")
        logger.info(f"median_translation_plane_error = {median_translation_plane_error}")
        logger.info(f"median_translation_line_error = {median_translation_line_error}")
        logger.info(f"median_rotation_error = {median_rotation_error}")
        logger.info(f"median_affordance_error = {median_affordance_error}")
        # max
        max_translation_distance_error = np.max(translation_distance_errors, axis=0)
        max_translation_along_error = np.max(translation_along_errors, axis=0)
        max_translation_perp_error = np.max(translation_perp_errors, axis=0)
        max_translation_plane_error = np.max(translation_plane_errors, axis=0)
        max_translation_line_error = np.max(translation_line_errors, axis=0)
        max_rotation_error = np.max(rotation_errors, axis=0)
        max_affordance_error = np.max(affordance_errors, axis=0)
        logger.info(f"max_translation_distance_error = {max_translation_distance_error}")
        logger.info(f"max_translation_along_error = {max_translation_along_error}")
        logger.info(f"max_translation_perp_error = {max_translation_perp_error}")
        logger.info(f"max_translation_plane_error = {max_translation_plane_error}")
        logger.info(f"max_translation_line_error = {max_translation_line_error}")
        logger.info(f"max_rotation_error = {max_rotation_error}")
        logger.info(f"max_affordance_error = {max_affordance_error}")
        # min
        min_translation_distance_error = np.min(translation_distance_errors, axis=0)
        min_translation_along_error = np.min(translation_along_errors, axis=0)
        min_translation_perp_error = np.min(translation_perp_errors, axis=0)
        min_translation_plane_error = np.min(translation_plane_errors, axis=0)
        min_translation_line_error = np.min(translation_line_errors, axis=0)
        min_rotation_error = np.min(rotation_errors, axis=0)
        min_affordance_error = np.min(affordance_errors, axis=0)
        logger.info(f"min_translation_distance_error = {min_translation_distance_error}")
        logger.info(f"min_translation_along_error = {min_translation_along_error}")
        logger.info(f"min_translation_perp_error = {min_translation_perp_error}")
        logger.info(f"min_translation_plane_error = {min_translation_plane_error}")
        logger.info(f"min_translation_line_error = {min_translation_line_error}")
        logger.info(f"min_rotation_error = {min_rotation_error}")
        logger.info(f"min_affordance_error = {min_affordance_error}")
        # std
        std_translation_distance_error = np.std(translation_distance_errors, axis=0)
        std_translation_along_error = np.std(translation_along_errors, axis=0)
        std_translation_perp_error = np.std(translation_perp_errors, axis=0)
        std_translation_plane_error = np.std(translation_plane_errors, axis=0)
        std_translation_line_error = np.std(translation_line_errors, axis=0)
        std_rotation_error = np.std(rotation_errors, axis=0)
        std_affordance_error = np.std(affordance_errors, axis=0)
        logger.info(f"std_translation_distance_error = {std_translation_distance_error}")
        logger.info(f"std_translation_along_error = {std_translation_along_error}")
        logger.info(f"std_translation_perp_error = {std_translation_perp_error}")
        logger.info(f"std_translation_plane_error = {std_translation_plane_error}")
        logger.info(f"std_translation_line_error = {std_translation_line_error}")
        logger.info(f"std_rotation_error = {std_rotation_error}")
        logger.info(f"std_affordance_error = {std_affordance_error}")
        # outliers
        translation_outliers_ratio = translation_outliers_num / data_num
        rotation_outliers_ratio = rotation_outliers_num / data_num
        affordance_outliers_ratio = affordance_outliers_num / data_num
        logger.info(f"translation_outliers_num = {translation_outliers_num}")
        logger.info(f"rotation_outliers_num = {rotation_outliers_num}")
        logger.info(f"affordance_outliers_num = {affordance_outliers_num}")
        logger.info(f"translation_outliers_ratio = {translation_outliers_ratio}")
        logger.info(f"rotation_outliers_ratio = {rotation_outliers_ratio}")
        logger.info(f"affordance_outliers_ratio = {affordance_outliers_ratio}")

        logger.info(f"data_num = {data_num}")

        if tb_writer is not None:
            tb_writer.add_scalars(f'{split}/translation_distance_error', {
                'mean': mean_translation_distance_error.item(), 
                'median': median_translation_distance_error.item(), 
                'max': max_translation_distance_error.item(), 
                'min': min_translation_distance_error.item(), 
                'std': std_translation_distance_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/translation_along_error', {
                'mean': mean_translation_along_error.item(), 
                'median': median_translation_along_error.item(), 
                'max': max_translation_along_error.item(), 
                'min': min_translation_along_error.item(), 
                'std': std_translation_along_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/translation_perp_error', {
                'mean': mean_translation_perp_error.item(), 
                'median': median_translation_perp_error.item(), 
                'max': max_translation_perp_error.item(), 
                'min': min_translation_perp_error.item(), 
                'std': std_translation_perp_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/translation_plane_error', {
                'mean': mean_translation_plane_error.item(), 
                'median': median_translation_plane_error.item(), 
                'max': max_translation_plane_error.item(), 
                'min': min_translation_plane_error.item(), 
                'std': std_translation_plane_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/translation_line_error', {
                'mean': mean_translation_line_error.item(), 
                'median': median_translation_line_error.item(), 
                'max': max_translation_line_error.item(), 
                'min': min_translation_line_error.item(), 
                'std': std_translation_line_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/rotation_error', {
                'mean': mean_rotation_error.item(), 
                'median': median_rotation_error.item(), 
                'max': max_rotation_error.item(), 
                'min': min_rotation_error.item(), 
                'std': std_rotation_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/affordance_error', {
                'mean': mean_affordance_error.item(), 
                'median': median_affordance_error.item(), 
                'max': max_affordance_error.item(), 
                'min': min_affordance_error.item(), 
                'std': std_affordance_error.item()
            }, epoch)
            tb_writer.add_scalars(f'{split}/outliers_ratio', {
                'translation': translation_outliers_ratio, 
                'rotation': rotation_outliers_ratio, 
                'affordance': affordance_outliers_ratio
            }, epoch)
        else:
            pass
    elif len(translation_distance_errors.shape) == 2:
        data_num = translation_distance_errors.shape[0]
        joint_num = translation_distance_errors.shape[1]
        if 'names' in results_dict.keys():
            names = results_dict['names']
            for j in range(joint_num):
                dataframe = pd.DataFrame({'name': names, 
                                          'translation_distance_errors': translation_distance_errors[:, j], 
                                          'translation_along_errors': translation_along_errors[:, j], 
                                          'translation_perp_errors': translation_perp_errors[:, j], 
                                          'translation_plane_errors': translation_plane_errors[:, j], 
                                          'translation_line_errors': translation_line_errors[:, j], 
                                          'rotation_errors': rotation_errors[:, j], 
                                          'affordance_errors': affordance_errors[:, j]})
                dataframe.to_csv(os.path.join(log_path, f'metrics{j}.csv'), sep=',')
        else:
            pass
        
        # mean
        mean_translation_distance_error = np.mean(translation_distance_errors, axis=0)
        mean_translation_along_error = np.mean(translation_along_errors, axis=0)
        mean_translation_perp_error = np.mean(translation_perp_errors, axis=0)
        mean_translation_plane_error = np.mean(translation_plane_errors, axis=0)
        mean_translation_line_error = np.mean(translation_line_errors, axis=0)
        mean_rotation_error = np.mean(rotation_errors, axis=0)
        mean_affordance_error = np.mean(affordance_errors, axis=0)
        logger.info(f"mean_translation_distance_error = {mean_translation_distance_error}")
        logger.info(f"mean_translation_along_error = {mean_translation_along_error}")
        logger.info(f"mean_translation_perp_error = {mean_translation_perp_error}")
        logger.info(f"mean_translation_plane_error = {mean_translation_plane_error}")
        logger.info(f"mean_translation_line_error = {mean_translation_line_error}")
        logger.info(f"mean_rotation_error = {mean_rotation_error}")
        logger.info(f"mean_affordance_error = {mean_affordance_error}")
        # median
        median_translation_distance_error = np.median(translation_distance_errors, axis=0)
        median_translation_along_error = np.median(translation_along_errors, axis=0)
        median_translation_perp_error = np.median(translation_perp_errors, axis=0)
        median_translation_plane_error = np.median(translation_plane_errors, axis=0)
        median_translation_line_error = np.median(translation_line_errors, axis=0)
        median_rotation_error = np.median(rotation_errors, axis=0)
        median_affordance_error = np.median(affordance_errors, axis=0)
        logger.info(f"median_translation_distance_error = {median_translation_distance_error}")
        logger.info(f"median_translation_along_error = {median_translation_along_error}")
        logger.info(f"median_translation_perp_error = {median_translation_perp_error}")
        logger.info(f"median_translation_plane_error = {median_translation_plane_error}")
        logger.info(f"median_translation_line_error = {median_translation_line_error}")
        logger.info(f"median_rotation_error = {median_rotation_error}")
        logger.info(f"median_affordance_error = {median_affordance_error}")
        # max
        max_translation_distance_error = np.max(translation_distance_errors, axis=0)
        max_translation_along_error = np.max(translation_along_errors, axis=0)
        max_translation_perp_error = np.max(translation_perp_errors, axis=0)
        max_translation_plane_error = np.max(translation_plane_errors, axis=0)
        max_translation_line_error = np.max(translation_line_errors, axis=0)
        max_rotation_error = np.max(rotation_errors, axis=0)
        max_affordance_error = np.max(affordance_errors, axis=0)
        logger.info(f"max_translation_distance_error = {max_translation_distance_error}")
        logger.info(f"max_translation_along_error = {max_translation_along_error}")
        logger.info(f"max_translation_perp_error = {max_translation_perp_error}")
        logger.info(f"max_translation_plane_error = {max_translation_plane_error}")
        logger.info(f"max_translation_line_error = {max_translation_line_error}")
        logger.info(f"max_rotation_error = {max_rotation_error}")
        logger.info(f"max_affordance_error = {max_affordance_error}")
        # min
        min_translation_distance_error = np.min(translation_distance_errors, axis=0)
        min_translation_along_error = np.min(translation_along_errors, axis=0)
        min_translation_perp_error = np.min(translation_perp_errors, axis=0)
        min_translation_plane_error = np.min(translation_plane_errors, axis=0)
        min_translation_line_error = np.min(translation_line_errors, axis=0)
        min_rotation_error = np.min(rotation_errors, axis=0)
        min_affordance_error = np.min(affordance_errors, axis=0)
        logger.info(f"min_translation_distance_error = {min_translation_distance_error}")
        logger.info(f"min_translation_along_error = {min_translation_along_error}")
        logger.info(f"min_translation_perp_error = {min_translation_perp_error}")
        logger.info(f"min_translation_plane_error = {min_translation_plane_error}")
        logger.info(f"min_translation_line_error = {min_translation_line_error}")
        logger.info(f"min_rotation_error = {min_rotation_error}")
        logger.info(f"min_affordance_error = {min_affordance_error}")
        # std
        std_translation_distance_error = np.std(translation_distance_errors, axis=0)
        std_translation_along_error = np.std(translation_along_errors, axis=0)
        std_translation_perp_error = np.std(translation_perp_errors, axis=0)
        std_translation_plane_error = np.std(translation_plane_errors, axis=0)
        std_translation_line_error = np.std(translation_line_errors, axis=0)
        std_rotation_error = np.std(rotation_errors, axis=0)
        std_affordance_error = np.std(affordance_errors, axis=0)
        logger.info(f"std_translation_distance_error = {std_translation_distance_error}")
        logger.info(f"std_translation_along_error = {std_translation_along_error}")
        logger.info(f"std_translation_perp_error = {std_translation_perp_error}")
        logger.info(f"std_translation_plane_error = {std_translation_plane_error}")
        logger.info(f"std_translation_line_error = {std_translation_line_error}")
        logger.info(f"std_rotation_error = {std_rotation_error}")
        logger.info(f"std_affordance_error = {std_affordance_error}")
        # outliers
        translation_outliers_ratio = translation_outliers_num / (data_num * joint_num)
        rotation_outliers_ratio = rotation_outliers_num / (data_num * joint_num)
        affordance_outliers_ratio = affordance_outliers_num / (data_num * joint_num)
        logger.info(f"translation_outliers_num = {translation_outliers_num}")
        logger.info(f"rotation_outliers_num = {rotation_outliers_num}")
        logger.info(f"affordance_outliers_num = {affordance_outliers_num}")
        logger.info(f"translation_outliers_ratio = {translation_outliers_ratio}")
        logger.info(f"rotation_outliers_ratio = {rotation_outliers_ratio}")
        logger.info(f"affordance_outliers_ratio = {affordance_outliers_ratio}")

        logger.info(f"data_num = {data_num}, joint_num = {joint_num}")

        if tb_writer is not None:
            for j in range(joint_num):
                tb_writer.add_scalars(f'{split}/joint_{j}/translation_distance_error', {
                    'mean': mean_translation_distance_error[j], 
                    'median': median_translation_distance_error[j], 
                    'max': max_translation_distance_error[j], 
                    'min': min_translation_distance_error[j], 
                    'std': std_translation_distance_error[j]
                }, epoch)
                tb_writer.add_scalars(f'{split}/joint_{j}/translation_along_error', {
                    'mean': mean_translation_along_error[j], 
                    'median': median_translation_along_error[j], 
                    'max': max_translation_along_error[j], 
                    'min': min_translation_along_error[j], 
                    'std': std_translation_along_error[j]
                }, epoch)
                tb_writer.add_scalars(f'{split}/joint_{j}/translation_perp_error', {
                    'mean': mean_translation_perp_error[j], 
                    'median': median_translation_perp_error[j], 
                    'max': max_translation_perp_error[j], 
                    'min': min_translation_perp_error[j], 
                    'std': std_translation_perp_error[j]
                }, epoch)
                tb_writer.add_scalars(f'{split}/joint_{j}/translation_plane_error', {
                    'mean': mean_translation_plane_error[j], 
                    'median': median_translation_plane_error[j], 
                    'max': max_translation_plane_error[j], 
                    'min': min_translation_plane_error[j], 
                    'std': std_translation_plane_error[j]
                }, epoch)
                tb_writer.add_scalars(f'{split}/joint_{j}/translation_line_error', {
                    'mean': mean_translation_line_error[j], 
                    'median': median_translation_line_error[j], 
                    'max': max_translation_line_error[j], 
                    'min': min_translation_line_error[j], 
                    'std': std_translation_line_error[j]
                }, epoch)
                tb_writer.add_scalars(f'{split}/joint_{j}/rotation_error', {
                    'mean': mean_rotation_error[j], 
                    'median': median_rotation_error[j], 
                    'max': max_rotation_error[j], 
                    'min': min_rotation_error[j], 
                    'std': std_rotation_error[j]
                }, epoch)
                tb_writer.add_scalars(f'{split}/joint_{j}/affordance_error', {
                    'mean': mean_affordance_error[j], 
                    'median': median_affordance_error[j], 
                    'max': max_affordance_error[j], 
                    'min': min_affordance_error[j], 
                    'std': std_affordance_error[j]
                }, epoch)
            tb_writer.add_scalars(f'{split}/outliers_ratio', {
                'translation': translation_outliers_ratio, 
                'rotation': rotation_outliers_ratio, 
                'affordance': affordance_outliers_ratio
            }, epoch)
        else:
            pass
    else:
        raise ValueError(f"Invalid shape of translation_distance_errors: {translation_distance_errors.shape}")
