import hydra
from omegaconf import DictConfig
import logging
import tqdm
import time
import numpy as np
import torch
from sklearn.cluster import KMeans

from datasets.rconfmask_afford_point_tuple_dataset import ArticulationDataset
from inference import voting_translation, voting_rotation
from utilities.metrics_utils import calc_translation_error, calc_translation_error_batch, calc_direction_error, calc_direction_error_batch, log_metrics
from utilities.vis_utils import visualize, visualize_translation_voting, visualize_rotation_voting
from utilities.env_utils import setup_seed
from utilities.constants import seed, light_blue_color, red_color, dark_red_color, dark_green_color, yellow_color


@hydra.main(config_path='./configs', config_name='test_gt_config', version_base='1.2')
def test_gt(cfg:DictConfig) -> None:
    logger = logging.getLogger('test_gt')
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    setup_seed(seed=cfg.general.seed)

    # prepare dataset
    logger.info("Preparing dataset...")
    device = cfg.general.device
    path = cfg.dataset.path
    categories = cfg.dataset.categories
    joint_num = cfg.dataset.joint_num
    resolution = cfg.dataset.resolution
    receptive_field = cfg.dataset.receptive_field
    denoise = cfg.dataset.denoise
    normalize = cfg.dataset.normalize
    sample_points_num = cfg.dataset.sample_points_num
    sample_tuples_num = cfg.algorithm.sample_tuples_num
    tuple_more_num = cfg.algorithm.tuple_more_num
    dataset = ArticulationDataset(path, categories, joint_num, resolution, receptive_field, 
                                  sample_points_num, sample_tuples_num, tuple_more_num, 
                                  rgb=False, denoise=denoise, normalize=normalize, 
                                  debug=False, vis=False, is_train=False)
    batch_size = cfg.general.batch_size
    num_workers = cfg.general.num_workers
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logger.info("Prepared dataset.")
    
    # test
    logger.info("GT Testing...")
    translation2pc = cfg.algorithm.translation2pc                           # solve translation voting too much far problem, but maybe sacrifice precision
    rotation_cluster = cfg.algorithm.rotation_cluster                       # solve rotation voting opposite problem, but still cannot solve it completely since maybe only 1 or 2 candidates
    if rotation_cluster:
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto')
    else:
        kmeans = None
    debug = False
    multi_candidate = cfg.algorithm.multi_candidate                         # solve translation and rotation voting discrete problem
    candidate_threshold = cfg.algorithm.candidate_threshold
    rotation_multi_neighbor = cfg.algorithm.rotation_multi_neighbor         # solve rotation voting discrete resolution problem
    neighbor_threshold = cfg.algorithm.neighbor_threshold
    angle_tol = cfg.algorithm.angle_tol                                     # solve rotation voting discrete resolution problem, but introduce voting opposite problem instead
    rot_candidate_num = int(4 * np.pi / (angle_tol / 180 * np.pi))
    voting_num = cfg.algorithm.voting_num
    bmm_size = cfg.algorithm.bmm_size
    with torch.no_grad():
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
        test_gt_start_time = time.time()
        for pcs, pc_normals, pc_shots, joint_translations, joint_rotations, affordable_positions, targets_tr, targets_rot, targets_afford, targets_conf, point_idxs_all in tqdm.tqdm(dataloader):
            # (B, N, 3), (B, N, 3), (B, N, 352), (B, J, 3), (B, J, 3), (B, J, 3), (B, J, N_t, 2), (B, J, N_t), (B, J, N_t, 2), (B, J, N_t), (B, N_t, 2 + N_m)
            B = pcs.shape[0]
            N_t = targets_tr.shape[2]
            batch_pred_translations, batch_pred_rotations, batch_pred_affordances = [], [], []
            for b in range(B):
                pc = pcs[b].numpy().astype(np.float32)                                          # (N, 3)
                # pc_normal = pc_normals[b].numpy().astype(np.float32)                            # (N, 3)
                # pc_shot = pc_shots[b].numpy().astype(np.float32)                                # (N, 352)
                joint_translation = joint_translations[b].numpy().astype(np.float32)            # (J, 3)
                joint_rotation = joint_rotations[b].numpy().astype(np.float32)                  # (J, 3)
                affordable_position = affordable_positions[b].numpy().astype(np.float32)        # (J, 3)
                target_tr = targets_tr[b].numpy().astype(np.float32)                            # (J, N_t, 2)
                target_rot = targets_rot[b].numpy().astype(np.float32)                          # (J, N_t)
                target_afford = targets_afford[b].numpy().astype(np.float32)                    # (J, N_t, 2)
                target_conf = targets_conf[b].numpy().astype(np.float32)                        # (J, N_t)
                point_idx_all = point_idxs_all[b].numpy().astype(np.int32)                      # (N_t, 2 + N_m)

                # inference
                pred_translations, pred_rotations, pred_affordances = [], [], []
                for j in range(joint_num):
                    this_target_tr = target_tr[j]                                               # (N_t, 2)
                    this_target_rot = target_rot[j]                                             # (N_t,)
                    this_target_afford = target_afford[j]                                       # (N_t, 2)
                    this_target_conf = target_conf[j]                                           # (N_t,)

                    pred_translation, grid_obj, corners = voting_translation(pc, this_target_tr, point_idx_all[:, :2], this_target_conf, 
                                                                             resolution, voting_num, device, 
                                                                             translation2pc, multi_candidate, candidate_threshold)
                    pred_translations.append(pred_translation)

                    pred_direction, sphere_pts, counts = voting_rotation(pc, this_target_rot, point_idx_all[:, :2], this_target_conf, 
                                                                         rot_candidate_num, angle_tol, voting_num, bmm_size, device, 
                                                                         multi_candidate, candidate_threshold, rotation_cluster, kmeans, 
                                                                         rotation_multi_neighbor, neighbor_threshold)
                    pred_rotations.append(pred_direction)

                    pred_affordance, agrid_obj, acorners = voting_translation(pc, this_target_afford, point_idx_all[:, :2], this_target_conf, 
                                                                              resolution, voting_num, device, 
                                                                              translation2pc, multi_candidate, candidate_threshold)
                    pred_affordances.append(pred_affordance)

                    translation_errors = calc_translation_error(pred_translation, joint_translation[j], pred_direction, joint_rotation[j])
                    if sum(translation_errors) > 20:
                        translation_outliers.append(translation_errors)
                    if debug and sum(translation_errors) > 20:
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
                    if debug and direction_error > 5:
                        print(f"{direction_error = }")
                        visualize_rotation_voting(sphere_pts, counts, pc, pc_color=light_blue_color, 
                                                  gt_rotation=joint_rotation[j], gt_color=dark_green_color, 
                                                  pred_rotation=pred_direction, pred_color=yellow_color, 
                                                  show_threshold=candidate_threshold, whether_frame=True, whether_bbox=True, window_name='rot_voting')
                        import pdb; pdb.set_trace()
                    affordance_error, _, _, _, _ = calc_translation_error(pred_affordance, affordable_position[j], None, None)
                    if affordance_error > 5:
                        affordance_outliers.append(affordance_error)
                    if debug and affordance_error > 5:
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
                    print(np.sum(this_target_conf), np.sum(this_target_conf) / N_t, sum(translation_errors), direction_error, affordance_error)
                # if debug:
                #     visualize(pc, pc_color=light_blue_color, pc_normal=pc_normal, 
                #               joint_translations=np.array(pred_translations), joint_rotations=np.array(pred_rotations), affordable_positions=np.array(pred_affordances), 
                #               joint_axis_colors=red_color, joint_point_colors=dark_red_color, 
                #               whether_frame=True, whether_bbox=True, window_name='pred')
                #     import pdb; pdb.set_trace()
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
            batch_affordance_errors, _, _, _, _ = calc_translation_error_batch(batch_pred_affordances, batch_gt_affordances, None, None)   # (B, J)
            translation_distance_errors.append(batch_translation_errors[0])
            translation_along_errors.append(batch_translation_errors[1])
            translation_perp_errors.append(batch_translation_errors[2])
            translation_plane_errors.append(batch_translation_errors[3])
            translation_line_errors.append(batch_translation_errors[4])
            rotation_errors.append(batch_rotation_errors)
            affordance_errors.append(batch_affordance_errors)
        test_gt_end_time = time.time()
        translation_distance_errors = np.concatenate(translation_distance_errors, axis=0)
        translation_along_errors = np.concatenate(translation_along_errors, axis=0)
        translation_perp_errors = np.concatenate(translation_perp_errors, axis=0)
        translation_plane_errors = np.concatenate(translation_plane_errors, axis=0)
        translation_line_errors = np.concatenate(translation_line_errors, axis=0)
        rotation_errors = np.concatenate(rotation_errors, axis=0)
        affordance_errors = np.concatenate(affordance_errors, axis=0)
        
        results_dict = {
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
        log_metrics(results_dict, logger, output_dir, tb_writer=None)
    logger.info(f"Time: {test_gt_end_time - test_gt_start_time}")
    logger.info("GT Tested.")


if __name__ == '__main__':
    test_gt()
