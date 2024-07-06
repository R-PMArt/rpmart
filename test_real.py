from typing import Dict, Union
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import time
import tqdm
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from datasets.point_tuple_dataset import ArticulationDataset
from models.roartnet import create_shot_encoder, create_encoder
from inference import voting_translation, voting_rotation
from utilities.metrics_utils import calc_translation_error, calc_translation_error_batch, calc_direction_error, calc_direction_error_batch, log_metrics
from utilities.vis_utils import visualize, visualize_translation_voting, visualize_rotation_voting, visualize_confidence_voting
from utilities.env_utils import setup_seed
from utilities.constants import seed, light_blue_color, red_color, dark_red_color, dark_green_color, yellow_color


def test_fn(test_dataloader:torch.utils.data.DataLoader, has_rgb:bool, shot_encoder:nn.Module, encoder:nn.Module, 
            resolution:float, voting_num:int, rot_bin_num:int, angle_tol:float, 
            translation2pc:bool, multi_candidate:bool, candidate_threshold:float, rotation_cluster:bool, 
            rotation_multi_neighbor:bool, neighbor_threshold:float, 
            bmm_size:int, test_num:int, device:int, vis:bool=False) -> Dict[str, Union[np.ndarray, int]]:
    if rotation_cluster:
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto')
    else:
        kmeans = None
    rot_candidate_num = int(4 * np.pi / (angle_tol / 180 * np.pi))
    
    tested_num = 0
    with torch.no_grad():
        names = []
        translation_distance_errors = []
        translation_along_errors = []
        translation_perp_errors = []
        translation_plane_errors = []
        translation_line_errors = []
        translation_outliers = []
        rotation_errors = []
        rotation_outliers = []
        affordance_errors = []
        affordance_outliers = []
        for batch_data in tqdm.tqdm(test_dataloader):
            if tested_num >= test_num:
                break
            if has_rgb:
                pcs, pc_normals, pc_shots, pc_colors, joint_translations, joint_rotations, affordable_positions, joint_types, point_idxs_all, batch_names = batch_data
                pcs, pc_normals, pc_shots, pc_colors, point_idxs_all = \
                    pcs.cuda(device), pc_normals.cuda(device), pc_shots.cuda(device), pc_colors.cuda(device), point_idxs_all.cuda(device)
            else:
                pcs, pc_normals, pc_shots, joint_translations, joint_rotations, affordable_positions, joint_types, point_idxs_all, batch_names = batch_data
                pcs, pc_normals, pc_shots, point_idxs_all = \
                    pcs.cuda(device), pc_normals.cuda(device), pc_shots.cuda(device), point_idxs_all.cuda(device)
            # (B, N, 3), (B, N, 3), (B, N, 352)(, (B, N, 3)), (B, J, 3), (B, J, 3), (B, J, 3), (B, J), (B, N_t, 2 + N_m), (B,)
            B = pcs.shape[0]
            N = pcs.shape[1]
            J = joint_translations.shape[1]
            N_t = point_idxs_all.shape[1]
            tested_num += B
            
            # shot encoder for every point
            shot_feat = shot_encoder(pc_shots)       # (B, N, N_s)
            
            # encoder for sampled point tuples
            # shot_inputs = torch.cat([shot_feat[point_idxs_all[:, i]] for i in range(0, point_idxs_all.shape[-1])], -1)        # (sample_points, feature_dim * (2 + num_more))
            # normal_inputs = torch.cat([torch.max(torch.sum(normal[point_idxs_all[:, i]] * normal[point_idxs_all[:, j]], dim=-1, keepdim=True), 
            #                                      torch.sum(-normal[point_idxs_all[:, i]] * normal[point_idxs_all[:, j]], dim=-1, keepdim=True))
            #                                      for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1)    # (sample_points, (2+num_more \choose 2))
            # coord_inputs = torch.cat([pc[point_idxs_all[:, i]] - pc[point_idxs_all[:, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1) # (sample_points, 3 * (2+num_more \choose 2))
            # shot_inputs = []
            # normal_inputs = []
            # coord_inputs = []
            # for b in range(pcs.shape[0]):
            #     shot_inputs.append(torch.cat([shot_feat[b][point_idxs_all[b, :, i]] for i in range(0, point_idxs_all.shape[-1])], dim=-1))    # (sample_points, feature_dim * (2 + num_more))
            #     normal_inputs.append(torch.cat([torch.max(torch.sum(normals[b][point_idxs_all[b, :, i]] * normals[b][point_idxs_all[b, :, j]], dim=-1, keepdim=True), 
            #                                      torch.sum(-normals[b][point_idxs_all[b, :, i]] * normals[b][point_idxs_all[b, :, j]], dim=-1, keepdim=True))
            #                                      for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1))   # (sample_points, (2+num_more \choose 2))
            #     coord_inputs.append(torch.cat([pcs[b][point_idxs_all[b, :, i]] - pcs[b][point_idxs_all[b, :, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1)) # (sample_points, 3 * (2+num_more \choose 2))
            # shot_inputs = torch.stack(shot_inputs, dim=0)     # (B, sample_points, feature_dim * (2 + num_more))
            # normal_inputs = torch.stack(normal_inputs, dim=0) # (B, sample_points, (2+num_more \choose 2))
            # coord_inputs = torch.stack(coord_inputs, dim=0)   # (B, sample_points, 3 * (2+num_more \choose 2))
            shot_inputs = torch.cat([
                torch.gather(shot_feat, 1, 
                            point_idxs_all[:, :, i:i+1].expand(
                            (B, N_t, shot_feat.shape[-1]))) 
                for i in range(point_idxs_all.shape[-1])], dim=-1)      # (B, N_t, N_s * (2 + N_m))
            normal_inputs = torch.cat([torch.max(
                torch.sum(torch.gather(pc_normals, 1, 
                                    point_idxs_all[:, :, i:i+1].expand(
                                    (B, N_t, pc_normals.shape[-1]))) * 
                        torch.gather(pc_normals, 1, 
                                    point_idxs_all[:, :, j:j+1].expand(
                                    (B, N_t, pc_normals.shape[-1]))), 
                        dim=-1, keepdim=True), 
                torch.sum(-torch.gather(pc_normals, 1, 
                                    point_idxs_all[:, :, i:i+1].expand(
                                    (B, N_t, pc_normals.shape[-1]))) * 
                        torch.gather(pc_normals, 1, 
                                    point_idxs_all[:, :, j:j+1].expand(
                                    (B, N_t, pc_normals.shape[-1]))), 
                        dim=-1, keepdim=True)) 
                for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1)     # (B, N_t, (2+N_m \choose 2))
            coord_inputs = torch.cat([
                torch.gather(pcs, 1, 
                            point_idxs_all[:, :, i:i+1].expand(
                            (B, N_t, pcs.shape[-1]))) - 
                torch.gather(pcs, 1, 
                            point_idxs_all[:, :, j:j+1].expand(
                            (B, N_t, pcs.shape[-1]))) 
                for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1)    # (B, N_t, 3 * (2+N_m \choose 2))
            if has_rgb:
                rgb_inputs = torch.cat([
                        torch.gather(pc_colors, 1, 
                                     point_idxs_all[:, :, i:i+1].expand(
                                     (B, N_t, pc_colors.shape[-1]))) 
                        for i in range(point_idxs_all.shape[-1])], dim=-1)      # (B, N_t, 3 * (2 + N_m))
                inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs, rgb_inputs], dim=-1)
            else:
                inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], dim=-1)
            preds = encoder(inputs)                     # (B, N_t, (2 + N_r + 2 + 1) * J)

            # voting
            batch_pred_translations, batch_pred_rotations, batch_pred_affordances = [], [], []
            pcs_numpy = pcs.cpu().numpy().astype(np.float32)                                    # (B, N, 3)
            pc_normals_numpy = pc_normals.cpu().numpy().astype(np.float32)                      # (B, N, 3)
            joint_translations_numpy = joint_translations.numpy().astype(np.float32)            # (B, J, 3)
            joint_rotations_numpy = joint_rotations.numpy().astype(np.float32)                  # (B, J, 3)
            affordable_positions_numpy = affordable_positions.numpy().astype(np.float32)        # (B, J, 3)
            point_idxs_numpy = point_idxs_all[:, :, :2].cpu().numpy().astype(np.int32)          # (B, N_t, 2)
            preds_numpy = preds.cpu().numpy().astype(np.float32)                                # (B, N_t, (2 + N_r + 2 + 1) * J)
            for b in range(B):
                pc = pcs_numpy[b]                                                               # (N, 3)
                pc_normal = pc_normals_numpy[b]                                                 # (N, 3)
                joint_translation = joint_translations_numpy[b]                                 # (J, 3)
                joint_rotation = joint_rotations_numpy[b]                                       # (J, 3)
                affordable_position = affordable_positions_numpy[b]                             # (J, 3)
                point_idx = point_idxs_numpy[b]                                                 # (N_t, 2)
                pred = preds_numpy[b]                                                           # (N_t, (2 + N_r + 2 + 1) * J)
                pred_tensor = torch.from_numpy(pred)
                
                pred_translations, pred_rotations, pred_affordances = [], [], []
                for j in range(J):
                    # conf selection
                    pred_conf = torch.sigmoid(pred_tensor[:, -1*J+j])                           # (N_t,)
                    not_selected_indices = pred_conf < 0.5
                    pred_conf[not_selected_indices] = 0
                    # pred_conf[pred_conf > 0] = 1
                    pred_conf = pred_conf.numpy()
                    if vis:
                        visualize_confidence_voting(pred_conf, pc, point_idx, 
                                                    whether_frame=True, whether_bbox=True, window_name='conf_voting')
                        import pdb; pdb.set_trace()

                    # translation voting
                    pred_tr = pred[:, 2*j:2*(j+1)]                                              # (N_t, 2)
                    pred_translation, grid_obj, corners = voting_translation(pc, pred_tr, point_idx, pred_conf, 
                                                                             resolution, voting_num, device, 
                                                                             translation2pc, multi_candidate, candidate_threshold)
                    pred_translations.append(pred_translation)
                    
                    # rotation voting
                    pred_rot = pred_tensor[:, (2*J+rot_bin_num*j):(2*J+rot_bin_num*(j+1))]      # (N_t, rot_bin_num)
                    pred_rot = torch.softmax(pred_rot, dim=-1)
                    pred_rot = torch.multinomial(pred_rot, 1).float()[:, 0]                     # (N_t,)
                    pred_rot = pred_rot / (rot_bin_num - 1) * np.pi
                    pred_rot = pred_rot.numpy()
                    pred_direction, sphere_pts, counts = voting_rotation(pc, pred_rot, point_idx, pred_conf, 
                                                                         rot_candidate_num, angle_tol, voting_num, bmm_size, device, 
                                                                         multi_candidate, candidate_threshold, rotation_cluster, kmeans, 
                                                                         rotation_multi_neighbor, neighbor_threshold)
                    pred_rotations.append(pred_direction)

                    # affordance voting
                    pred_afford = pred[:, (2*J+rot_bin_num*J+2*j):(2*J+rot_bin_num*J+2*(j+1))]  # (N_t, 2)
                    pred_affordance, agrid_obj, acorners = voting_translation(pc, pred_afford, point_idx, pred_conf, 
                                                                              resolution, voting_num, device, 
                                                                              translation2pc, multi_candidate, candidate_threshold)
                    pred_affordances.append(pred_affordance)

                    translation_errors = calc_translation_error(pred_translation, joint_translation[j], pred_direction, joint_rotation[j])
                    if sum(translation_errors) > 20:
                        translation_outliers.append(translation_errors)
                    if vis and sum(translation_errors) > 20:
                        print(f"{translation_errors = }")
                        indices = np.indices(grid_obj.shape)
                        indices_list = np.transpose(indices, (1, 2, 3, 0)).reshape(-1, len(grid_obj.shape))
                        votes_list = grid_obj.reshape(-1)
                        grid_pc = corners[0] + indices_list * resolution
                        visualize_translation_voting(grid_pc, votes_list, pc, pc_color=light_blue_color, 
                                                     gt_translation=joint_translation[j], gt_color=dark_green_color, 
                                                     pred_translation=pred_translation, pred_color=yellow_color, 
                                                     show_threshold=candidate_threshold, whether_frame=True, whether_bbox=True, window_name='tr_voting')
                        import pdb; pdb.set_trace()
                    direction_error = calc_direction_error(pred_direction, joint_rotation[j])
                    if direction_error > 5:
                        rotation_outliers.append(direction_error)
                    if vis and direction_error > 5:
                        print(f"{direction_error = }")
                        visualize_rotation_voting(sphere_pts, counts, pc, pc_color=light_blue_color, 
                                                  gt_rotation=joint_rotation[j], gt_color=dark_green_color, 
                                                  pred_rotation=pred_direction, pred_color=yellow_color, 
                                                  show_threshold=candidate_threshold, whether_frame=True, whether_bbox=True, window_name='rot_voting')
                        import pdb; pdb.set_trace()
                    affordance_error, _, _, _, _ = calc_translation_error(pred_affordance, affordable_position[j], None, None)
                    if affordance_error > 5:
                        affordance_outliers.append(affordance_error)
                    if vis and affordance_error > 5:
                        print(f"{affordance_error = }")
                        indices = np.indices(agrid_obj.shape)
                        indices_list = np.transpose(indices, (1, 2, 3, 0)).reshape(-1, len(agrid_obj.shape))
                        votes_list = agrid_obj.reshape(-1)
                        grid_pc = acorners[0] + indices_list * resolution
                        visualize_translation_voting(grid_pc, votes_list, pc, pc_color=light_blue_color, 
                                                     gt_translation=affordable_position[j], gt_color=dark_green_color, 
                                                     pred_translation=pred_affordance, pred_color=yellow_color, 
                                                     show_threshold=candidate_threshold, whether_frame=True, whether_bbox=True, window_name='afford_voting')
                        import pdb; pdb.set_trace()
                if vis:
                    visualize(pc, pc_color=light_blue_color, pc_normal=pc_normal, 
                              joint_translations=np.array(pred_translations), joint_rotations=np.array(pred_rotations), affordable_positions=np.array(pred_affordances), 
                              joint_axis_colors=red_color, joint_point_colors=dark_red_color, affordable_position_colors=dark_green_color, 
                              whether_frame=True, whether_bbox=True, window_name='pred')
                    import pdb; pdb.set_trace()
                batch_pred_translations.append(pred_translations)
                batch_pred_rotations.append(pred_rotations)
                batch_pred_affordances.append(pred_affordances)
            batch_pred_translations = np.array(batch_pred_translations).astype(np.float32)      # (B, J, 3)
            batch_pred_rotations = np.array(batch_pred_rotations).astype(np.float32)            # (B, J, 3)
            batch_pred_affordances = np.array(batch_pred_affordances).astype(np.float32)        # (B, J, 3)
            batch_gt_translations = joint_translations.numpy().astype(np.float32)               # (B, J, 3)
            batch_gt_rotations = joint_rotations.numpy().astype(np.float32)                     # (B, J, 3)
            batch_gt_affordances = affordable_positions.numpy().astype(np.float32)              # (B, J, 3)
            batch_translation_errors = calc_translation_error_batch(batch_pred_translations, batch_gt_translations, batch_pred_rotations, batch_gt_rotations)   # (B, J)
            batch_rotation_errors = calc_direction_error_batch(batch_pred_rotations, batch_gt_rotations)    # (B, J)
            batch_affordance_errors, _, _, _, _ = calc_translation_error_batch(batch_pred_affordances, batch_gt_affordances, None, None)    # (B, J)
            translation_distance_errors.append(batch_translation_errors[0])
            translation_along_errors.append(batch_translation_errors[1])
            translation_perp_errors.append(batch_translation_errors[2])
            translation_plane_errors.append(batch_translation_errors[3])
            translation_line_errors.append(batch_translation_errors[4])
            rotation_errors.append(batch_rotation_errors)
            affordance_errors.append(batch_affordance_errors)
            names.extend(batch_names)
        translation_distance_errors = np.concatenate(translation_distance_errors, axis=0)       # (tested_num, J)
        translation_along_errors = np.concatenate(translation_along_errors, axis=0)             # (tested_num, J)
        translation_perp_errors = np.concatenate(translation_perp_errors, axis=0)               # (tested_num, J)
        translation_plane_errors = np.concatenate(translation_plane_errors, axis=0)             # (tested_num, J)
        translation_line_errors = np.concatenate(translation_line_errors, axis=0)               # (tested_num, J)
        rotation_errors = np.concatenate(rotation_errors, axis=0)                               # (tested_num, J)
        affordance_errors = np.concatenate(affordance_errors, axis=0)                           # (tested_num, J)
    
    return {
        'names': names, 
        'translation_distance_errors': translation_distance_errors, 
        'translation_along_errors': translation_along_errors, 
        'translation_perp_errors': translation_perp_errors, 
        'translation_plane_errors': translation_plane_errors, 
        'translation_line_errors': translation_line_errors, 
        'translation_outliers_num': len(translation_outliers), 
        'rotation_errors': rotation_errors, 
        'rotation_outliers_num': len(rotation_outliers), 
        'affordance_errors': affordance_errors, 
        'affordance_outliers_num': len(affordance_outliers)
    }


@hydra.main(config_path='./configs', config_name='test_real_config', version_base='1.2')
def test_real(cfg:DictConfig) -> None:
    logger = logging.getLogger('test_real')
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    setup_seed(seed=cfg.testing.seed)
    trained_path = cfg.trained.path
    trained_cfg = OmegaConf.load(f"{trained_path}/.hydra/config.yaml")
    # merge trained_cfg into cfg, cfg has higher priority
    cfg = OmegaConf.merge(trained_cfg, cfg)
    print(OmegaConf.to_yaml(cfg))

    # prepare dataset
    logger.info("Preparing dataset...")
    device = cfg.testing.device
    path = cfg.dataset.path
    instances = cfg.dataset.instances
    joint_num = cfg.dataset.joint_num
    resolution = cfg.dataset.resolution
    receptive_field = cfg.dataset.receptive_field
    rgb = cfg.dataset.rgb
    denoise = cfg.dataset.denoise
    normalize = cfg.dataset.normalize
    sample_points_num = cfg.dataset.sample_points_num
    sample_tuples_num = cfg.algorithm.sampling.sample_tuples_num
    tuple_more_num = cfg.algorithm.sampling.tuple_more_num
    dataset = ArticulationDataset(path, instances, joint_num, resolution, receptive_field, 
                                  sample_points_num, sample_tuples_num, tuple_more_num, 
                                  rgb, denoise, normalize, debug=False, vis=False, is_train=False)
    
    batch_size = cfg.testing.batch_size
    num_workers = cfg.testing.num_workers
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info("Prepared dataset.")

    # prepare model
    logger.info("Preparing model...")
    shot_hidden_dims = cfg.algorithm.shot_encoder.hidden_dims
    shot_feature_dim = cfg.algorithm.shot_encoder.feature_dim
    shot_bn = cfg.algorithm.shot_encoder.bn
    shot_ln = cfg.algorithm.shot_encoder.ln
    shot_droput = cfg.algorithm.shot_encoder.dropout
    shot_encoder = create_shot_encoder(shot_hidden_dims, shot_feature_dim, 
                                       shot_bn, shot_ln, shot_droput)
    shot_encoder.load_state_dict(torch.load(f'{os.path.join(trained_path, "weights")}/shot_encoder_latest.pth', map_location=torch.device(device)))
    shot_encoder = shot_encoder.cuda(device)
    overall_hidden_dims = cfg.algorithm.encoder.hidden_dims
    rot_bin_num = cfg.algorithm.voting.rot_bin_num
    overall_bn = cfg.algorithm.encoder.bn
    overall_ln = cfg.algorithm.encoder.ln
    overall_dropout = cfg.algorithm.encoder.dropout
    encoder = create_encoder(tuple_more_num, shot_feature_dim, rgb, overall_hidden_dims, rot_bin_num, joint_num, 
                             overall_bn, overall_ln, overall_dropout)
    encoder.load_state_dict(torch.load(f'{os.path.join(trained_path, "weights")}/encoder_latest.pth', map_location=torch.device(device)))
    encoder = encoder.cuda(device)
    logger.info("Prepared model.")

    # testing
    voting_num = cfg.algorithm.voting.voting_num
    angle_tol = cfg.algorithm.voting.angle_tol
    translation2pc = cfg.algorithm.voting.translation2pc
    multi_candidate = cfg.algorithm.voting.multi_candidate
    candidate_threshold = cfg.algorithm.voting.candidate_threshold
    rotation_multi_neighbor = cfg.algorithm.voting.rotation_multi_neighbor
    neighbor_threshold = cfg.algorithm.voting.neighbor_threshold
    rotation_cluster = cfg.algorithm.voting.rotation_cluster
    bmm_size = cfg.algorithm.voting.bmm_size
    logger.info("Testing...")
    testing_start_time = time.time()
    shot_encoder.eval()
    encoder.eval()

    testing_results = test_fn(dataloader, rgb, shot_encoder, encoder, 
                              resolution, voting_num, rot_bin_num, angle_tol, 
                              translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                              rotation_multi_neighbor, neighbor_threshold, 
                              bmm_size, len(dataset), device, vis=cfg.vis)
    log_metrics(testing_results, logger, output_dir, tb_writer=None)

    testing_end_time = time.time()
    logger.info("Tested.")
    logger.info("Testing time: " + str(testing_end_time - testing_start_time))


if __name__ == '__main__':
    test_real()
