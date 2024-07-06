import os
import configargparse
from omegaconf import OmegaConf
import time
import numpy as np
import torch

from utilities.env_utils import setup_seed
from utilities.data_utils import transform_pc, transform_dir
from utilities.metrics_utils import invaffordance_metrics, invaffordances2affordance
from utilities.constants import seed, max_grasp_width


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # data config
    parser.add_argument('--cat', type=str, default='Microwave', help='the category of the object')
    # model config
    parser.add_argument('--roartnet', action='store_true', help='whether call roartnet')
    parser.add_argument('--roartnet_config_path', type=str, default='./configs/eval_config.yaml', help='the path to roartnet config')
    # grasp config
    parser.add_argument('--graspnet', action='store_true', help='whether call graspnet')
    parser.add_argument('--gsnet_weight_path', type=str, default='./weights/checkpoint_detection.tar', help='the path to graspnet weight')
    parser.add_argument('--max_grasp_width', type=float, default=max_grasp_width, help='the max width of the gripper')
    # task config
    parser.add_argument('--selected_part', type=int, default=0, help='the selected part of the object')
    # others
    parser.add_argument('--seed', type=int, default=seed, help='the random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("please clear temporary data directory")

    args = config_parse()
    setup_seed(args.seed)
    temp_request_path = './temp_data/observation.npz'
    temp_response_path = './temp_data/service.npz'
    temp_flag_path = './temp_data/flag.npy'
    if args.cat == "Microwave":
        joint_types = [0]
        joint_res = [-1]
    elif args.cat == "Refrigerator":
        joint_types = [0]
        joint_res = [1]
    elif args.cat == "Safe":
        joint_types = [0]
        joint_res = [1]
    elif args.cat == "StorageFurniture":
        joint_types = [1, 0]
        joint_res = [0, -1]
    elif args.cat == "Drawer":
        joint_types = [1, 1, 1]
        joint_res = [0, 0, 0]
    elif args.cat == "WashingMachine":
        joint_types = [0]
        joint_res = [-1]
    else:
        raise ValueError(f"Unknown category {args.cat}")

    if args.roartnet:
        print("===> loading roartnet")
        start_time = time.time()
        from models.roartnet import create_shot_encoder, create_encoder
        from inference import inference_fn as roartnet_inference_fn
        roartnet_cfg = OmegaConf.load(args.roartnet_config_path)
        trained_path = roartnet_cfg.trained.path[args.cat]
        trained_cfg = OmegaConf.load(f"{trained_path}/.hydra/config.yaml")
        roartnet_cfg = OmegaConf.merge(trained_cfg, roartnet_cfg)
        joint_num = roartnet_cfg.dataset.joint_num
        resolution = roartnet_cfg.dataset.resolution
        receptive_field = roartnet_cfg.dataset.receptive_field
        has_rgb = roartnet_cfg.dataset.rgb
        denoise = roartnet_cfg.dataset.denoise
        normalize = roartnet_cfg.dataset.normalize
        sample_points_num = roartnet_cfg.dataset.sample_points_num
        sample_tuples_num = roartnet_cfg.algorithm.sampling.sample_tuples_num
        tuple_more_num = roartnet_cfg.algorithm.sampling.tuple_more_num
        shot_hidden_dims = roartnet_cfg.algorithm.shot_encoder.hidden_dims
        shot_feature_dim = roartnet_cfg.algorithm.shot_encoder.feature_dim
        shot_bn = roartnet_cfg.algorithm.shot_encoder.bn
        shot_ln = roartnet_cfg.algorithm.shot_encoder.ln
        shot_dropout = roartnet_cfg.algorithm.shot_encoder.dropout
        shot_encoder = create_shot_encoder(shot_hidden_dims, shot_feature_dim, 
                                           shot_bn, shot_ln, shot_dropout)
        shot_encoder.load_state_dict(torch.load(f'{trained_path}/weights/shot_encoder_latest.pth', map_location=torch.device('cuda')))
        shot_encoder = shot_encoder.cuda()
        shot_encoder.eval()
        overall_hidden_dims = roartnet_cfg.algorithm.encoder.hidden_dims
        rot_bin_num = roartnet_cfg.algorithm.voting.rot_bin_num
        overall_bn = roartnet_cfg.algorithm.encoder.bn
        overall_ln = roartnet_cfg.algorithm.encoder.ln
        overall_dropout = roartnet_cfg.algorithm.encoder.dropout
        encoder = create_encoder(tuple_more_num, shot_feature_dim, has_rgb, overall_hidden_dims, rot_bin_num, joint_num, 
                                overall_bn, overall_ln, overall_dropout)
        encoder.load_state_dict(torch.load(f'{trained_path}/weights/encoder_latest.pth', map_location=torch.device('cuda')))
        encoder = encoder.cuda()
        encoder.eval()
        voting_num = roartnet_cfg.algorithm.voting.voting_num
        angle_tol = roartnet_cfg.algorithm.voting.angle_tol
        translation2pc = roartnet_cfg.algorithm.voting.translation2pc
        multi_candidate = roartnet_cfg.algorithm.voting.multi_candidate
        candidate_threshold = roartnet_cfg.algorithm.voting.candidate_threshold
        rotation_multi_neighbor = roartnet_cfg.algorithm.voting.rotation_multi_neighbor
        neighbor_threshold = roartnet_cfg.algorithm.voting.neighbor_threshold
        rotation_cluster = roartnet_cfg.algorithm.voting.rotation_cluster
        bmm_size = roartnet_cfg.algorithm.voting.bmm_size
        end_time = time.time()
        print(f"===> loaded roartnet {end_time - start_time}")

    if args.graspnet:
        print("===> loading graspnet")
        start_time = time.time()
        from munch import DefaultMunch
        from gsnet import AnyGrasp
        grasp_detector_cfg = {
            'checkpoint_path': args.gsnet_weight_path, 
            'max_gripper_width': args.max_grasp_width, 
            'gripper_height': 0.03, 
            'top_down_grasp': False, 
            'add_vdistance': True, 
            'debug': True
        }
        grasp_detector_cfg = DefaultMunch.fromDict(grasp_detector_cfg)
        grasp_detector = AnyGrasp(grasp_detector_cfg)
        grasp_detector.load_net()
        end_time = time.time()
        print(f"===> loaded graspnet {end_time - start_time}")
    
    while True:
        print("===> listening to request")
        start_time = time.time()
        serviced = np.array(False)
        np.save(temp_flag_path, serviced)
        got_request = False
        while not got_request:
            got_request = os.path.exists(temp_request_path)
            time.sleep(5.0)             # NOTE: hardcode to be longer than the writing time
            if got_request:
                while not (os.path.isfile(temp_request_path) and os.access(temp_request_path, os.R_OK)):
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        observation = np.load(temp_request_path, allow_pickle=True)
        cam_pc = observation['point_cloud']
        pc_rgb = observation['rgb']
        c2c = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        cam_pc_model = transform_pc(cam_pc, c2c)
        time.sleep(0.5)
        os.remove(temp_request_path)
        end_time = time.time()
        print(f"===> got request {end_time - start_time}")

        print("===> inferencing model")
        if args.roartnet:
            start_time = time.time()
            pred_joint_bases, pred_joint_directions, pred_affordable_positions = roartnet_inference_fn(cam_pc_model, pc_rgb if has_rgb else None, shot_encoder, encoder, 
                                                                                                       denoise, normalize, resolution, receptive_field, sample_points_num, sample_tuples_num, tuple_more_num, 
                                                                                                       voting_num, rot_bin_num, angle_tol, 
                                                                                                       translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                                                                                                       rotation_multi_neighbor, neighbor_threshold, bmm_size, joint_num, device=0)
            pred_selected_joint_base = pred_joint_bases[args.selected_part]
            pred_selected_joint_direction = pred_joint_directions[args.selected_part]
            pred_selected_affordable_position = pred_affordable_positions[args.selected_part]
            pred_selected_joint_base = transform_pc(pred_selected_joint_base[None, :], np.linalg.inv(c2c))[0]
            pred_selected_joint_direction = transform_dir(pred_selected_joint_direction[None, :], np.linalg.inv(c2c))[0]
            pred_selected_affordable_position = transform_pc(pred_selected_affordable_position[None, :], np.linalg.inv(c2c))[0]
            end_time = time.time()
            print(f"===> shot_dropout predicted {end_time - start_time}")
        
        print("===> detecting grasps")
        if args.graspnet:
            start_time = time.time()
            # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=None, lims=[
            #     np.floor(np.min(pcd_grasp[:, 0])) - 0.1, np.ceil(np.max(pcd_grasp[:, 0])) + 0.1, 
            #     np.floor(np.min(pcd_grasp[:, 1])) - 0.1, np.ceil(np.max(pcd_grasp[:, 1])) + 0.1, 
            #     np.floor(np.min(pcd_grasp[:, 2])) - 0.1, np.ceil(np.max(pcd_grasp[:, 2])) + 0.1])
            # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=[-float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf')])
            # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=None, apply_object_mask=True, dense_grasp=False, collision_detection=True)
            try:
                gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='fast')
            except:
                gg_grasp = grasp_detector.get_grasp(cam_pc.astype(np.float32), colors=pc_rgb, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='slow')
            if gg_grasp is None:
                gg_grasp = []
            else:
                if len(gg_grasp) != 2:
                    gg_grasp = []
                else:
                    gg_grasp, pcd_o3d = gg_grasp
                    gg_grasp = gg_grasp.nms().sort_by_score()
                    # grippers_o3d = gg_grasp.to_open3d_geometry_list()
                    # frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    # o3d.visualization.draw_geometries([*grippers_o3d, pcd_o3d, frame_o3d])
            if len(gg_grasp) == 0:
                np.savez(temp_response_path, 
                         joint_base=pred_selected_joint_base, 
                         joint_direction=pred_selected_joint_direction, 
                         affordable_position=pred_selected_affordable_position, 
                         joint_type=joint_types[args.selected_part], 
                         joint_re=joint_res[args.selected_part], 
                         num_grasps=0)
                while not (os.path.isfile(temp_response_path) and os.access(temp_response_path, os.R_OK)):
                    time.sleep(0.1)
                serviced = np.array(True)
                np.save(temp_flag_path, serviced)
                while True:
                    while not (os.path.isfile(temp_flag_path) and os.access(temp_flag_path, os.R_OK)):
                        time.sleep(0.1)
                    serviced = np.load(temp_flag_path).item()
                    if not serviced:
                        os.remove(temp_flag_path)
                        os.remove(temp_response_path)
                        break
                    else:
                        time.sleep(0.1)
                continue
            grasp_scores, grasp_widths, grasp_depths, grasp_translations, grasp_rotations, grasp_invaffordances = [], [], [], [], [], []
            for g_idx, g_grasp in enumerate(gg_grasp):
                grasp_score = g_grasp.score
                grasp_scores.append(grasp_score)
                grasp_width = g_grasp.width
                grasp_widths.append(grasp_width)
                grasp_depth = g_grasp.depth
                grasp_depths.append(grasp_depth)
                grasp_translation = g_grasp.translation
                grasp_translations.append(grasp_translation)
                grasp_rotation = g_grasp.rotation_matrix
                grasp_rotations.append(grasp_rotation)
                grasp_invaffordance = invaffordance_metrics(grasp_translation, grasp_rotation, grasp_score, pred_selected_affordable_position, 
                                                            pred_selected_joint_base, pred_selected_joint_direction, joint_types[args.selected_part])
                grasp_invaffordances.append(grasp_invaffordance)
            grasp_affordances = invaffordances2affordance(grasp_invaffordances)
            selected_grasp_idx = np.argmax(grasp_affordances)
            selected_grasp_score = grasp_scores[selected_grasp_idx]
            selected_grasp_width = grasp_widths[selected_grasp_idx]
            selected_grasp_width = max(min(selected_grasp_width * 1.5, args.max_grasp_width), 0.0)
            selected_grasp_depth = grasp_depths[selected_grasp_idx]
            selected_grasp_translation = grasp_translations[selected_grasp_idx]
            selected_grasp_rotation = grasp_rotations[selected_grasp_idx]
            selected_grasp_affordance = grasp_affordances[selected_grasp_idx]
            end_time = time.time()
            print(f"===> anygrasp detected {end_time - start_time} {len(gg_grasp)}")
        
        print("===> sending response")
        start_time = time.time()
        np.savez(temp_response_path, 
                 joint_base=pred_selected_joint_base, 
                 joint_direction=pred_selected_joint_direction, 
                 affordable_position=pred_selected_affordable_position, 
                 joint_type=joint_types[args.selected_part], 
                 joint_re=joint_res[args.selected_part], 
                 num_grasps=len(gg_grasp), 
                 grasp_score=selected_grasp_score, 
                 grasp_width=selected_grasp_width, 
                 grasp_depth=selected_grasp_depth, 
                 grasp_translation=selected_grasp_translation, 
                 grasp_rotation=selected_grasp_rotation, 
                 grasp_affordance=selected_grasp_affordance)
        while not (os.path.isfile(temp_response_path) and os.access(temp_response_path, os.R_OK)):
            time.sleep(0.1)
        serviced = np.array(True)
        np.save(temp_flag_path, serviced)
        while True:
            while not (os.path.isfile(temp_flag_path) and os.access(temp_flag_path, os.R_OK)):
                time.sleep(0.1)
            serviced = np.load(temp_flag_path).item()
            if not serviced:
                os.remove(temp_flag_path)
                os.remove(temp_response_path)
                break
            else:
                time.sleep(0.1)
        end_time = time.time()
        print(f"===> sent response {end_time - start_time}")
