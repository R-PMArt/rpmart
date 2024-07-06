import configargparse
import json
from omegaconf import OmegaConf
import random
import os
import logging
import numpy as np
import tqdm
import time
import datetime
import torch
import imageio
from PIL import Image
import transformations as tf
import open3d as o3d

from envs.env import Env
from envs.camera import Camera
from envs.robot import Robot
from utilities.env_utils import setup_seed
from utilities.data_utils import transform_pc, transform_dir, read_joints_from_urdf_file, pc_noise
from utilities.metrics_utils import calc_pose_error, invaffordance_metrics, invaffordances2affordance, calc_translation_error, calc_direction_error
from utilities.constants import seed, max_grasp_width


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    # environment config
    parser.add_argument('--camera_config_path', type=str, default='./configs/data/camera_config.json', help='the path to camera config')
    parser.add_argument('--num_config_per_object', type=int, default=200, help='the number of configs per object')
    # data config
    parser.add_argument('--scale', type=float, default=0, help='the scale of the object')
    parser.add_argument('--data_path', type=str, default='/data2/junbo/where2act_modified_sapien_dataset/7167/mobility_vhacd.urdf', help='the path to the data')
    parser.add_argument('--cat', type=str, default='Microwave', help='the category of the object')
    parser.add_argument('--object_config_path', type=str, default='./configs/data/object_config.json', help='the path to object config')
    # robot config
    parser.add_argument('--robot_urdf_path', type=str, default='/data2/junbo/franka_panda/panda_gripper.urdf', help='the path to robot urdf')
    parser.add_argument('--robot_scale', type=float, default=1, help='the scale of the robot')
    # gt config
    parser.add_argument('--gt_path', type=str, default='/data2/junbo/where2act_modified_sapien_dataset/7167/joint_abs_pose.json', help='the path to gt')
    # model config
    parser.add_argument('--roartnet', action='store_true', help='whether call roartnet')
    parser.add_argument('--roartnet_config_path', type=str, default='./configs/eval_config.yaml', help='the path to roartnet config')
    # grasp config
    parser.add_argument('--graspnet', action='store_true', help='whether call graspnet')
    parser.add_argument('--grasp', action='store_true', help='whether grasp')
    parser.add_argument('--gsnet_weight_path', type=str, default='./weights/checkpoint_detection.tar', help='the path to graspnet weight')
    parser.add_argument('--show_all_grasps', action='store_true', help='whether show all grasps detected by graspnet, note the visuals will harm the speed')
    # task config
    parser.add_argument('--selected_part', type=int, default=0, help='the selected part of the object')
    # parser.add_argument('--manipulation', type=float, default=-20, help='the manipulation task, positive as push, negative as pull, revolute in degree, prismatic in cm')
    parser.add_argument('--task', type=str, default='pull', choices=['pull', 'push', 'none'], help='the task')
    parser.add_argument('--task_low', type=float, default=0.1, help='low bound of task')
    parser.add_argument('--task_high', type=float, default=0.7, help='high bound of task')
    parser.add_argument('--success_threshold', type=float, default=0.15, help='success threshold for ratio of manipulated movement')
    # others
    parser.add_argument('--gui', action='store_true', help='whether show gui')
    parser.add_argument('--video', action='store_true', help='whether save video')
    parser.add_argument('--output_path', type=str, default='./outputs/manipulation', help='the path to output')
    parser.add_argument('--abbr', type=str, default='7167', help='the abbr of the object')
    parser.add_argument('--seed', type=int, default=seed, help='the random seed')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = config_parse()
    setup_seed(args.seed)
    # TODO: hardcode here to add noise to point cloud
    output_name = 'noise_' + args.cat + '_' + args.abbr + '_' + args.task
    if args.roartnet:
        output_name += '_roartnet'
    output_name += '_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_path = os.path.join(args.output_path, output_name)
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger("manipulation")
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(filename=os.path.join(output_path, 'log.txt'))
    logger.addHandler(handler)

    if args.roartnet:
        # load roartnet
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
        shot_droput = roartnet_cfg.algorithm.shot_encoder.dropout
        shot_encoder = create_shot_encoder(shot_hidden_dims, shot_feature_dim, 
                                           shot_bn, shot_ln, shot_droput)
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
        logger.info(f"===> loaded roartnet {end_time - start_time}")
    
    if args.graspnet:
        # load graspnet
        start_time = time.time()
        from munch import DefaultMunch
        from gsnet import AnyGrasp
        grasp_detector_cfg = {
            'checkpoint_path': args.gsnet_weight_path, 
            'max_gripper_width': max_grasp_width * args.robot_scale, 
            'gripper_height': 0.03, 
            'top_down_grasp': False, 
            'add_vdistance': True, 
            'debug': True
        }
        grasp_detector_cfg = DefaultMunch.fromDict(grasp_detector_cfg)
        grasp_detector = AnyGrasp(grasp_detector_cfg)
        grasp_detector.load_net()
        end_time = time.time()
        logger.info(f"===> loaded graspnet {end_time - start_time}")

    # initialize environment
    start_time = time.time()
    env = Env(show_gui=args.gui)
    if args.camera_config_path == 'none':
        camera_config = None
        cam = Camera(env, random_position=True, restrict_dir=True)
    else:
        with open(args.camera_config_path, "r") as fp:
            camera_config = json.load(fp)
        camera_near = camera_config['intrinsics']['near']
        camera_far = camera_config['intrinsics']['far']
        camera_width = camera_config['intrinsics']['width']
        camera_height = camera_config['intrinsics']['height']
        camera_fovx = np.random.uniform(camera_config['intrinsics']['fovx'][0], camera_config['intrinsics']['fovx'][1])
        camera_fovy = np.random.uniform(camera_config['intrinsics']['fovy'][0], camera_config['intrinsics']['fovy'][1])
        camera_dist = np.random.uniform(camera_config['extrinsics']['dist'][0], camera_config['extrinsics']['dist'][1])
        camera_phi = np.random.uniform(camera_config['extrinsics']['phi'][0], camera_config['extrinsics']['phi'][1])
        camera_theta = np.random.uniform(camera_config['extrinsics']['theta'][0], camera_config['extrinsics']['theta'][1])
        cam = Camera(env, near=camera_near, far=camera_far, image_size=[camera_width, camera_height], fov=[camera_fovx, camera_fovy], 
                        dist=camera_dist, phi=camera_phi / 180 * np.pi, theta=camera_theta / 180 * np.pi)
    if args.gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
    object_material = env.get_material(4, 4, 0.01)
    if args.object_config_path == 'none':
        object_config = {
            "scale_min": 1, 
            "scale_max": 1, 
        }
    else:
        with open(args.object_config_path, "r") as fp:
            object_config = json.load(fp)
    if args.gt_path != "none":
        with open(args.gt_path, "r") as fp:
            gt_config = json.load(fp)
        max_object_size = 0
        for joint_name in gt_config.keys():
            if joint_name == 'aabb_min' or joint_name == 'aabb_max':
                continue
            max_object_size = max(max_object_size, np.max(np.array(gt_config[joint_name]["bbox_max"]) - np.array(gt_config[joint_name]["bbox_min"])))
        if args.cat in object_config:
            object_scale = object_config[args.cat]['size'] / max_object_size
            joint_num = object_config[args.cat]['joint_num']
            assert joint_num == len(gt_config.keys()) - 2
        else:
            object_scale = 1
            joint_num = len(gt_config.keys()) - 2
        if args.scale != 0:
            object_scale = args.scale
    else:
        object_scale = args.scale
    end_time = time.time()
    logger.info(f"===> initialized environment {end_time - start_time}")
    
    success_configs, fail_configs = [], []
    for config_id in tqdm.trange(args.num_config_per_object):
        # load object
        this_config_scale = np.random.uniform(object_config["scale_min"], object_config["scale_max"]) * object_scale
        video = []
        start_time = time.time()
        still = False
        try_times = 0
        while not still and try_times < 5:
            goal_qpos, joint_abs_angles = env.load_object(args.data_path, object_material, state='random-middle-middle', target_part_id=-1, target_part_idx=args.selected_part, scale=this_config_scale)
            env.render()

            # check still and reach goal qpos
            start_time = time.time()
            still_timesteps = 0
            wait_timesteps = 0
            cur_qpos = env.get_object_qpos()
            # while still_timesteps < 500 and wait_timesteps < 3000:
            while still_timesteps < 500 and wait_timesteps < 5000:
                env.step()
                env.render()
                cur_new_qpos = env.get_object_qpos()
                invalid_contact = False
                for c in env.scene.get_contacts():
                    for p in c.points:
                        if abs(p.impulse @ p.impulse) > 1e-4:
                            invalid_contact = True
                            break
                    if invalid_contact:
                        break
                # if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
                if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and np.max(np.abs(cur_new_qpos - goal_qpos)) < 0.02 and (not invalid_contact):
                    still_timesteps += 1
                else:
                    still_timesteps = 0
                cur_qpos = cur_new_qpos
                wait_timesteps += 1
            still = still_timesteps >= 500
            if not still:
                env.scene.remove_articulation(env.object)
            try_times += 1
        if not still:
            logger.info(f"{config_id} failed to load object")
            continue
        end_time = time.time()
        logger.info(f"===> {config_id} loaded object {end_time - start_time} {this_config_scale} {try_times}")

        # set camera
        if camera_config is None:
            cam.change_pose(random_position=True, restrict_dir=True)
        else:
            camera_fovx = np.random.uniform(camera_config['intrinsics']['fovx'][0], camera_config['intrinsics']['fovx'][1])
            camera_fovy = np.random.uniform(camera_config['intrinsics']['fovy'][0], camera_config['intrinsics']['fovy'][1])
            cam.change_fov([camera_fovx, camera_fovy])
            camera_dist = np.random.uniform(camera_config['extrinsics']['dist'][0], camera_config['extrinsics']['dist'][1])
            camera_phi = np.random.uniform(camera_config['extrinsics']['phi'][0], camera_config['extrinsics']['phi'][1])
            camera_theta = np.random.uniform(camera_config['extrinsics']['theta'][0], camera_config['extrinsics']['theta'][1])
            cam.change_pose(dist=camera_dist, phi=camera_phi / 180 * np.pi, theta=camera_theta / 180 * np.pi)
        if args.gui:
            env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
        env.step()
        env.render()
        if args.video:
            frame, _ = cam.get_observation()
            frame = (frame * 255).astype(np.uint8)
            frame = Image.fromarray(frame)
            video.append(frame)
        
        # get observation
        rgb, depth = cam.get_observation()
        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        mask = cam.camera.get_segmentation()                    # used only for filtering out bad cases
        object_mask = mask[cam_XYZA_id1, cam_XYZA_id2]
        extrinsic = cam.get_metadata()['mat44']
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]
        pcd_cam = cam_XYZA_pts.copy()
        pcd_world = (R @ cam_XYZA_pts.T).T + T
        pcd_color = rgb[cam_XYZA_id1, cam_XYZA_id2]
        c2c = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        pcd_grasp = transform_pc(pcd_cam, c2c)
        # TODO: hardcode here to add noise to point cloud
        pcd_cam = pc_noise(pcd_cam, 0.2, 0.01, 0.002, 0.5)
        
        # obtain gt
        start_time = time.time()
        # env.add_frame_visual()
        # env.render()
        if args.gt_path != "none":
            # set task
            movable_link_ids = env.movable_link_ids
            object_joint_types = env.movable_link_joint_types
            movable_link_joint_names = env.movable_link_joint_names
            all_link_ids = env.all_link_ids
            all_link_names = env.all_link_names
            joint_real_states = env.get_object_qpos()
            joint_lower_states = env.joint_angles_lower
            joint_upper_states = env.joint_angles_upper
            env.set_target_object_part_actor_id(movable_link_ids[args.selected_part])
            if object_joint_types[args.selected_part] == 0:
                target_joint_real_state = joint_real_states[args.selected_part] / np.pi * 180
                target_joint_lower_state = joint_lower_states[args.selected_part] / np.pi * 180
                target_joint_upper_state = joint_upper_states[args.selected_part] / np.pi * 180
            elif object_joint_types[args.selected_part] == 1:
                target_joint_real_state = joint_real_states[args.selected_part] * 100
                target_joint_lower_state = joint_lower_states[args.selected_part] * 100
                target_joint_upper_state = joint_upper_states[args.selected_part] * 100
            else:
                raise ValueError(f"invalid joint type {object_joint_types[args.selected_part]}")
            if args.task == 'push':
                task_high = min(args.task_high * (target_joint_upper_state - target_joint_lower_state), target_joint_real_state - target_joint_lower_state - 15)
            elif args.task == 'pull':
                task_high = min(args.task_high * (target_joint_upper_state - target_joint_lower_state), target_joint_upper_state - target_joint_real_state - 5)
            elif args.task == 'none':
                task_high = target_joint_upper_state - target_joint_lower_state
            else:
                raise ValueError(f"invalid task {args.task}")
            task_low = args.task_low * (target_joint_upper_state - target_joint_lower_state)
            if task_high <= 0 or task_low >= task_high:
                logger.info(f"{config_id} cannot set task {target_joint_upper_state - target_joint_lower_state} {target_joint_real_state - target_joint_lower_state}")
                env.scene.remove_articulation(env.object)
                continue
            if args.task == 'push':
                task_state = random.random() * (task_high - task_low) + task_low
            elif args.task == 'pull':
                task_state = random.random() * (task_high - task_low) + task_low
                task_state *= -1
            elif args.task == 'none':
                task_state = 0
            else:
                raise ValueError(f"invalid task {args.task}")
            assert len(movable_link_ids) == len(object_joint_types) and len(object_joint_types) == len(movable_link_joint_names)
            assert len(movable_link_ids) == len(joint_real_states) and len(joint_real_states) == len(joint_lower_states) and len(joint_lower_states) == len(joint_upper_states)
            logger.info(f"{config_id} task {task_state}")

            joint_bases, joint_directions, joint_types, joint_res, joint_states, affordable_positions = [], [], [], [], [], []
            for idx, link_id in enumerate(movable_link_ids):
                joint_pose_meta = gt_config[movable_link_joint_names[idx]]
                joint_base = np.asarray(joint_pose_meta['base_position'], order='F') * this_config_scale
                joint_bases.append(joint_base)
                # env.add_point_visual(joint_base, color=[0, 1, 0], name='joint_base_{}'.format(idx))
                joint_direction = np.asarray(joint_pose_meta['direction'], order='F')
                joint_directions.append(joint_direction)
                # env.add_line_visual(joint_base, joint_base + this_config_scale * joint_direction, color=[0, 1, 0], name='joint_direction_{}'.format(idx))
                joint_type = joint_pose_meta['joint_type']
                assert joint_type == object_joint_types[idx]
                joint_types.append(joint_type)
                if joint_type == 0:
                    joint_re = joint_pose_meta["joint_re"]
                elif joint_type == 1:
                    joint_re = 0
                else:
                    raise ValueError(f"invalid joint type {joint_pose_meta['joint_type']}")
                joint_res.append(joint_re)
                joint_state = joint_real_states[idx] - joint_lower_states[idx]
                joint_states.append(joint_state)
                affordable_position = np.asarray(joint_pose_meta['affordable_position'], order='F') * this_config_scale
                if joint_type == 0:
                    transformation_matrix = tf.rotation_matrix(angle=joint_state * joint_re, direction=joint_direction, point=joint_base)
                elif joint_type == 1:
                    transformation_matrix = tf.translation_matrix(joint_state * joint_direction)
                else:
                    raise ValueError(f"invalid joint type {joint_pose_meta['joint_type']}")
                affordable_position = transform_pc(affordable_position[None, :], transformation_matrix)[0]
                affordable_positions.append(affordable_position)
                # env.add_point_visual(affordable_position, color=[0, 0, 1], name='affordable_position_{}'.format(idx))
                if joint_type == 0:
                    logger.info(f"added joint {movable_link_joint_names[idx]} revolute {'pull_counterclockwise' if joint_re == 1 else 'pull_clockwise'} {joint_state / np.pi * 180} {(joint_upper_states[idx] - joint_lower_states[idx]) / np.pi * 180}")
                elif joint_type == 1:
                    logger.info(f"added joint {movable_link_joint_names[idx]} prismatic {joint_state * 100} {(joint_upper_states[idx] - joint_lower_states[idx]) * 100}")
                else:
                    raise ValueError(f"invalid joint type {joint_pose_meta['joint_type']}")
            selected_joint_base = joint_bases[args.selected_part]
            selected_joint_direction = joint_directions[args.selected_part]
            selected_joint_type = joint_types[args.selected_part]
            selected_joint_re = joint_res[args.selected_part]
            selected_joint_state = joint_states[args.selected_part]
            selected_affordable_position = affordable_positions[args.selected_part]

            # merge affiliated parts into parents
            joints_dict = read_joints_from_urdf_file(args.data_path)
            link_graph = {}
            for joint_name in joints_dict:
                link_graph[joints_dict[joint_name]['child']] = joints_dict[joint_name]['parent']
            mask_ins = np.unique(object_mask)
            fixed_id = -1
            for mask_in in mask_ins:
                if mask_in not in movable_link_ids:
                    link_name = all_link_names[all_link_ids.index(mask_in)]
                    parent_link = link_graph[link_name]
                    parent_link_id = all_link_ids[all_link_names.index(parent_link)]
                    if parent_link_id in mask_ins:
                        if parent_link_id in movable_link_ids:
                            object_mask[object_mask == mask_in] = parent_link_id
                        else:
                            # TODO: may be error
                            if fixed_id == -1:
                                fixed_id = parent_link_id
                            object_mask[object_mask == mask_in] = fixed_id
                    else:
                        # TODO: may be error
                        if fixed_id == -1:
                            fixed_id = parent_link_id
                        object_mask[object_mask == mask_in] = fixed_id
            mask_ins = np.unique(object_mask)
            fixed_num = 0
            for mask_in in mask_ins:
                if mask_in not in movable_link_ids:
                    fixed_num += 1
            assert fixed_num <= 1
            assert mask_ins.shape[0] <= len(movable_link_ids) + 1

            # rearrange mask
            instance_mask = np.zeros_like(object_mask)
            real_id = 1
            selected_joint_idxs = []
            for mask_id, movable_id in zip(movable_link_ids, range(len(movable_link_ids))):
                if mask_id in mask_ins:
                    instance_mask[object_mask == mask_id] = real_id
                    real_id += 1
                    selected_joint_idxs.append(movable_id)

            # check regular
            if len(selected_joint_idxs) != joint_num or np.min(instance_mask) != 0 or np.max(instance_mask) != joint_num:
                logger.info(f"{config_id} irregular")
                env.scene.remove_articulation(env.object)
                continue
            
            # check seen
            if (instance_mask == (args.selected_part + 1)).sum() < 0.15 * instance_mask.shape[0]:
                logger.info(f"{config_id} unseen")
                env.scene.remove_articulation(env.object)
                continue

            # check suitable
            affordable_dist = np.linalg.norm(pcd_world - selected_affordable_position, axis=-1).min()
            if affordable_dist > 0.03:
                logger.info(f"{config_id} unsuitable")
                env.scene.remove_articulation(env.object)
                continue
            
            for _ in range(100):
                env.step()
                env.render()
        if args.video:
            frame, _ = cam.get_observation()
            frame = (frame * 255).astype(np.uint8)
            frame = Image.fromarray(frame)
            video.append(frame)
        end_time = time.time()
        logger.info(f"===> added gt {end_time - start_time}")

        # prediction
        if args.roartnet:
            start_time = time.time()
            pred_joint_bases, pred_joint_directions, pred_affordable_positions = roartnet_inference_fn(pcd_cam, pcd_color if has_rgb else None, shot_encoder, encoder, 
                                                                                                       denoise, normalize, resolution, receptive_field, sample_points_num, sample_tuples_num, tuple_more_num, 
                                                                                                       voting_num, rot_bin_num, angle_tol, 
                                                                                                       translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                                                                                                       rotation_multi_neighbor, neighbor_threshold, bmm_size, joint_num, device=0)
            pred_selected_joint_base = pred_joint_bases[args.selected_part]
            pred_selected_joint_direction = pred_joint_directions[args.selected_part]
            pred_selected_affordable_position = pred_affordable_positions[args.selected_part]
            pred_selected_joint_base = transform_pc(pred_selected_joint_base[None, :], extrinsic)[0]
            pred_selected_joint_direction = transform_dir(pred_selected_joint_direction[None, :], extrinsic)[0]
            pred_selected_affordable_position = transform_pc(pred_selected_affordable_position[None, :], extrinsic)[0]
            joint_translation_errors = calc_translation_error(pred_selected_joint_base, selected_joint_base, pred_selected_joint_direction, selected_joint_direction)
            joint_direction_error = calc_direction_error(pred_selected_joint_direction, selected_joint_direction)
            affordance_error, _, _, _, _ = calc_translation_error(pred_selected_affordable_position, selected_affordable_position, None, None)
            selected_joint_base = pred_selected_joint_base
            selected_joint_direction = pred_selected_joint_direction
            selected_affordable_position = pred_selected_affordable_position
            end_time = time.time()
            logger.info(f"===> {config_id} roartnet predicted {end_time - start_time} {joint_translation_errors} {joint_direction_error} {affordance_error}")
        
        # obtain grasp
        if args.graspnet:
            start_time = time.time()
            # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=None, lims=[
            #     np.floor(np.min(pcd_grasp[:, 0])) - 0.1, np.ceil(np.max(pcd_grasp[:, 0])) + 0.1, 
            #     np.floor(np.min(pcd_grasp[:, 1])) - 0.1, np.ceil(np.max(pcd_grasp[:, 1])) + 0.1, 
            #     np.floor(np.min(pcd_grasp[:, 2])) - 0.1, np.ceil(np.max(pcd_grasp[:, 2])) + 0.1])
            # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=[-float('inf'), float('inf'), -float('inf'), float('inf'), -float('inf'), float('inf')])
            # gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=None, apply_object_mask=True, dense_grasp=False, collision_detection=True)
            try:
                gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='fast')
            except:
                gg_grasp = grasp_detector.get_grasp(pcd_grasp.astype(np.float32), colors=pcd_color, lims=None, voxel_size=0.0075, apply_object_mask=False, dense_grasp=True, collision_detection='slow')
            if gg_grasp is None:
                gg_grasp = []
            else:
                if len(gg_grasp) != 2:
                    gg_grasp = []
                else:
                    gg_grasp, pcd_o3d = gg_grasp
                    gg_grasp = gg_grasp.nms().sort_by_score()
                    # if args.gui:
                    #     grippers_o3d = gg_grasp.to_open3d_geometry_list()
                    #     frame_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    #     o3d.visualization.draw_geometries([*grippers_o3d, pcd_o3d, frame_o3d])
            end_time = time.time()
            logger.info(f"===> {config_id} obtained grasp {end_time - start_time} {len(gg_grasp)}")
            if len(gg_grasp) == 0:
                logger.info(f"{config_id} no grasp detected")
                env.scene.remove_articulation(env.object)
                continue

            # add visuals
            start_time = time.time()
            grasp_scores, grasp_widths, grasp_depths, grasp_translations, grasp_rotations, grasp_invaffordances = [], [], [], [], [], []
            for g_idx, g_grasp in enumerate(gg_grasp):
                grasp_score = g_grasp.score
                grasp_scores.append(grasp_score)
                grasp_width = g_grasp.width
                grasp_widths.append(grasp_width)
                grasp_depth = g_grasp.depth
                grasp_depths.append(grasp_depth)
                grasp_translation = g_grasp.translation
                grasp_rotation = g_grasp.rotation_matrix
                grasp_transformation = np.identity(4)
                grasp_transformation[:3, :3] = grasp_rotation
                grasp_transformation[:3, 3] = grasp_translation
                grasp_transformation = extrinsic @ np.linalg.inv(c2c) @ grasp_transformation
                grasp_translation = grasp_transformation[:3, 3]
                grasp_translations.append(grasp_translation)
                grasp_rotation = grasp_transformation[:3, :3]
                grasp_rotations.append(grasp_rotation)
                grasp_invaffordance = invaffordance_metrics(grasp_translation, grasp_rotation, grasp_score, selected_affordable_position, 
                                                            selected_joint_base, selected_joint_direction, selected_joint_type)
                grasp_invaffordances.append(grasp_invaffordance)
            grasp_affordances = invaffordances2affordance(grasp_invaffordances)
            if args.show_all_grasps:
                for g_idx, g_grasp in enumerate(gg_grasp):
                    # env.add_grasp_visual(grasp_widths[g_idx], grasp_depths[g_idx], grasp_translations[g_idx], grasp_rotations[g_idx], affordance=(grasp_affordances[g_idx] - min(grasp_affordances)) / (max(grasp_affordances) - min(grasp_affordances)), name='grasp_{}'.format(g_idx))
                    print("added grasp", grasp_affordances[g_idx])
            selected_grasp_idx = np.argmax(grasp_affordances)
            selected_grasp_score = grasp_scores[selected_grasp_idx]
            selected_grasp_width = grasp_widths[selected_grasp_idx]
            selected_grasp_width = max(min(selected_grasp_width * 1.5, max_grasp_width * args.robot_scale), 0.0)
            selected_grasp_depth = grasp_depths[selected_grasp_idx]
            selected_grasp_translation = grasp_translations[selected_grasp_idx]
            selected_grasp_rotation = grasp_rotations[selected_grasp_idx]
            selected_grasp_affordance = grasp_affordances[selected_grasp_idx]
            # env.add_grasp_visual(selected_grasp_width, selected_grasp_depth, selected_grasp_translation, selected_grasp_rotation, affordance=selected_grasp_affordance, name='selected_grasp')
            selected_grasp_pose = np.identity(4)
            selected_grasp_pose[:3, :3] = selected_grasp_rotation
            selected_grasp_pose[:3, 3] = selected_grasp_translation
            for _ in range(100):
                env.step()
                env.render()
            if args.video:
                frame, _ = cam.get_observation()
                frame = (frame * 255).astype(np.uint8)
                frame = Image.fromarray(frame)
                video.append(frame)
            end_time = time.time()
            logger.info(f"===> {config_id} added visuals {end_time - start_time}")
        
            # load robot
            if args.grasp:
                start_time = time.time()
                robot_material = env.get_material(4, 4, 0.01)
                robot_move_steps_per_unit = 1000
                robot_short_wait_steps = 10
                robot_long_wait_steps = 1000
                robot = Robot(env, args.robot_urdf_path, robot_material, open_gripper=False, scale=args.robot_scale)
                env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)
                selected_grasp_pre_pose = selected_grasp_pose.copy()
                selected_grasp_pre_pose[:3, 3] -= 0.1 * selected_grasp_pre_pose[:3, 0]
                robot.set_gripper(selected_grasp_width)
                robot.set_pose(selected_grasp_pre_pose, selected_grasp_depth)
                # env.add_point_visual(selected_grasp_pre_pose[:3, 3], color=[1, 0, 0], radius=0.01, name='selected_grasp_pre_translation')
                for _ in range(100):
                    env.step()
                    env.render()
                if args.video:
                    frame, _ = cam.get_observation()
                    frame = (frame * 255).astype(np.uint8)
                    frame = Image.fromarray(frame)
                    video.append(frame)
                read_width = robot.get_gripper()
                read_pose = robot.get_pose(selected_grasp_depth)
                pose_error = calc_pose_error(selected_grasp_pre_pose, read_pose)
                end_time = time.time()
                logger.info(f"===> {config_id} loaded robot {end_time - start_time} {read_width - selected_grasp_width} {pose_error}")
                
                # grasp
                start_time = time.time()
                frames = robot.move_to_target_pose(selected_grasp_pose, selected_grasp_depth, robot_move_steps_per_unit * 10, vis_gif=args.video, cam=cam)
                if args.video:
                    video.extend(frames)
                frames = robot.wait_n_steps(robot_short_wait_steps, vis_gif=args.video, cam=cam)
                if args.video:
                    video.extend(frames)
                robot.close_gripper()
                frames = robot.wait_n_steps(robot_long_wait_steps, vis_gif=args.video, cam=cam)
                if args.video:
                    video.extend(frames)
                current_width = robot.get_gripper()
                success_grasp = env.check_contact_right()
                read_pose = robot.get_pose(selected_grasp_depth)
                pose_error = calc_pose_error(selected_grasp_pose, read_pose)
                # env.add_point_visual(selected_grasp_pose[:3, 3], color=[0, 0, 1], radius=0.01, name='selected_grasp_translation')
                # env.render()
                # if args.video:
                #     frame, _ = cam.get_observation()
                #     frame = (frame * 255).astype(np.uint8)
                #     frame = Image.fromarray(frame)
                #     video.append(frame)
                end_time = time.time()
                logger.info(f"===> {config_id} grasped {end_time - start_time} {pose_error} {success_grasp}")

                # manipulation
                if task_state != 0:
                    plan_trajectory, real_trajectory, semi_trajectory = [], [], []
                    plan_current_pose = robot.get_pose(selected_grasp_depth)
                    joint_state_initial = env.get_target_part_state()
                    if selected_joint_type == 0:
                        joint_state_initial = joint_state_initial / np.pi * 180
                    elif selected_joint_type == 1:
                        joint_state_initial = joint_state_initial * 100
                    else:
                        raise ValueError(f"invalid joint type {selected_joint_type}")
                    joint_state_task = joint_state_initial - task_state * (1 - args.success_threshold)
                    start_time = time.time()
                    for step in tqdm.trange(int(np.ceil(np.abs(task_state / 2.0) * 1.5))):
                        current_pose = robot.get_pose(selected_grasp_depth)
                        if selected_joint_type == 0:
                            rotation_angle = -2.0 * np.sign(task_state) * selected_joint_re / 180.0 * np.pi
                            delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=selected_joint_direction, point=selected_joint_base)
                        elif selected_joint_type == 1:
                            translation_distance = -2.0 * np.sign(task_state) / 100.0
                            delta_pose = tf.translation_matrix(selected_joint_direction * translation_distance)
                        else:
                            raise ValueError(f"invalid joint type {selected_joint_type}")
                        # next_pose = delta_pose @ plan_current_pose
                        next_pose = delta_pose @ current_pose
                        frames = robot.move_to_target_pose(next_pose, selected_grasp_depth, robot_move_steps_per_unit * 2, vis_gif=args.video, cam=cam)
                        if args.video:
                            video.extend(frames)
                        # robot.wait_n_steps(robot_short_wait_steps)
                        read_pose = robot.get_pose(selected_grasp_depth)
                        pose_error = calc_pose_error(next_pose, read_pose)
                        current_joint_state = env.get_target_part_state()
                        if selected_joint_type == 0:
                            current_joint_state = current_joint_state / np.pi * 180
                        elif selected_joint_type == 1:
                            current_joint_state = current_joint_state * 100
                        else:
                            raise ValueError(f"invalid joint type {selected_joint_type}")
                        joint_state_error = current_joint_state - joint_state_task
                        logger.info(f"{pose_error} {joint_state_error}")
                        real_trajectory.append(read_pose)
                        semi_trajectory.append(next_pose)
                        plan_next_pose = delta_pose @ plan_current_pose
                        plan_current_pose = plan_next_pose.copy()
                        plan_trajectory.append(plan_next_pose)
                        success_manipulation = ((task_state < 0) and (joint_state_error > 0)) or ((task_state > 0) and (joint_state_error < 0))
                        if success_manipulation:
                            break
                    # for step in tqdm.trange(int(np.ceil(np.abs(task_state / 2.0)))):
                    #     if step % 2 != 0:
                    #         continue
                    #     env.add_grasp_visual(current_width, selected_grasp_depth, plan_trajectory[step][:3, 3], plan_trajectory[step][:3, :3], affordance=1, name='grasp_{}'.format(-step))
                    #     env.add_grasp_visual(current_width, selected_grasp_depth, real_trajectory[step][:3, 3], real_trajectory[step][:3, :3], affordance=0, name='grasp_{}_real'.format(-step))
                    #     env.add_grasp_visual(current_width, selected_grasp_depth, semi_trajectory[step][:3, 3], semi_trajectory[step][:3, :3], affordance=0.99, name='grasp_{}_semi'.format(-step))
                    # env.render()
                    # if args.video:
                    #     frame, _ = cam.get_observation()
                    #     frame = (frame * 255).astype(np.uint8)
                    #     frame = Image.fromarray(frame)
                    #     video.append(frame)
                    end_time = time.time()
                    logger.info(f"===> {config_id} manipulated {end_time - start_time} {success_manipulation}")
                    if success_manipulation:
                        success_configs.append(config_id)
                    else:
                        fail_configs.append(config_id)
                env.scene.remove_articulation(robot.robot)
        env.scene.remove_articulation(env.object)
        
        if args.video:
            imageio.mimsave(os.path.join(output_path, f'video_{str(config_id).zfill(2)}_{str(success_manipulation)}.mp4'), video)
    
    logger.info(f"===> success configs {success_configs} {len(success_configs)}")
    logger.info(f"===> fail configs {fail_configs} {len(fail_configs)}")
    logger.info(f"===> success rate {len(success_configs) / (len(success_configs) + len(fail_configs))}")
