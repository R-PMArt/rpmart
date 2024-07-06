import os
import time
import tqdm
import numpy as np
import transformations as tf

from envs.real_camera import CameraL515
from envs.real_robot import Panda
from utilities.data_utils import transform_pc, transform_dir
from utilities.vis_utils import visualize
from utilities.network_utils import send, read


if __name__ == '__main__':
    print("please manually set the robot to a specific pose, make sure remote services all running")

    cam2EE_path = "/home/franka/junbo/data/robot/l515_franka.npy"
    cam2EE = np.load(cam2EE_path)
    camera_loaded = False
    robot_loaded = False
    vis = True
    temp_observation_path = "./temp_data/observation.npz"
    temp_service_path = "./temp_data/service.npz"
    temp_flag_path = "./temp_data/flag.npy"
    remote_repo_path = "TODO"                   # TODO: set your own remote repo path
    remote_observation_path = f"{remote_repo_path}/temp_data/observation.npz"
    remote_service_path = f"{remote_repo_path}/temp_data/service.npz"
    remote_flag_path = f"{remote_repo_path}/temp_data/flag.npy"
    remote_ip = "TODO"                          # TODO: set your own remote ip
    port = TODO                                 # TODO: set your own remote port
    username = "TODO"                           # TODO: set your own remote username
    key_filename = "TODO"                       # TODO: set your own local key file path
    task = -1               # close as 1, open as -1
    time_steps = 10

    try:
        print("===> initializing camera")
        start_time = time.time()
        camera = CameraL515()
        camera_loaded = True
        end_time = time.time()
        print("===> camera initialized", end_time - start_time)

        print("===> initializing robot")
        start_time = time.time()
        robot = Panda()
        robot.gripper_open()
        robot.homing()
        robot_loaded = True
        end_time = time.time()
        print("===> robot initialized", end_time - start_time)

        print("===> getting observation")
        start_time = time.time()
        color, depth = camera.get_data(hole_filling=False)
        depth_sensor = camera.pipeline_profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        xyzrgb = camera.getXYZRGB(color, depth, np.identity(4), np.identity(4), camera.getIntrinsics(), inpaint=False, depth_scale=depth_scale)
        # xyzrgb = xyzrgb[xyzrgb[:, 2] <= 1.5, :]
        xyzrgb = xyzrgb[xyzrgb[:, 2] > 0.05, :]
        cam_pc = xyzrgb[:, 0:3]
        pc_color = xyzrgb[:, 3:6]
        end_time = time.time()
        print("===> observation got", end_time - start_time)

        print("===> preprocessing observation")
        start_time = time.time()
        EE2robot = robot.readPose()
        cam2robot = EE2robot @ cam2EE
        robot2cam = np.linalg.inv(cam2robot)
        base_pc = transform_pc(cam_pc, cam2robot)
        space_mask_x = np.logical_and(base_pc[:, 0] > 0, base_pc[:, 0] < 1.1)
        space_mask_y = np.logical_and(base_pc[:, 1] > -0.27, base_pc[:, 1] < 0.55)
        # space_mask_z = base_pc[:, 2] > 0.02
        # space_mask_z = base_pc[:, 2] > 0.55             # microwave: pad + safe (rotate)
        # space_mask_z = base_pc[:, 2] > 0.52             # refrigerator: storagefurniture
        # space_mask_z = base_pc[:, 2] > 0.4              # safe: pad + microwave
        # space_mask_z = base_pc[:, 2] > 0.27             # storagefurniture: microwave
        # space_mask_z = base_pc[:, 2] > 0.27             # drawer: microwave
        space_mask_z = base_pc[:, 2] > 0.4              # washingmachine: pad + microwave
        space_mask = np.logical_and(np.logical_and(space_mask_x, space_mask_y), space_mask_z)
        base_pc_space = base_pc[space_mask, :]
        pc_color_space = pc_color[space_mask, :]
        cam_pc_space = transform_pc(base_pc_space, robot2cam)
        end_time = time.time()
        print("===> observation preprocessed", end_time - start_time)
        np.savez("./observation.npz", point_cloud=cam_pc_space, rgb=pc_color_space)
        if vis:
            visualize(cam_pc_space, pc_color_space, whether_frame=True, whether_bbox=True, window_name="observation")
        
        print("===> sending request")
        start_time = time.time()
        np.savez(temp_observation_path, point_cloud=cam_pc_space, rgb=pc_color_space)
        time.sleep(0.5)
        while not (os.path.isfile(temp_observation_path) and os.access(temp_observation_path, os.R_OK)):
            time.sleep(0.1)
        send(temp_observation_path, remote_observation_path, 
             remote_ip=remote_ip, port=port, username=username, key_filename=key_filename)
        time.sleep(0.5)
        os.remove(temp_observation_path)
        end_time = time.time()
        print("===> request sent", end_time - start_time)

        print("===> reading response")
        start_time = time.time()
        while True:
            read(temp_flag_path, remote_flag_path, 
                 remote_ip=remote_ip, port=port, username=username, key_filename=key_filename)
            time.sleep(0.5)
            got_service = np.load(temp_flag_path).item()
            if got_service:
                os.remove(temp_flag_path)
                break
            else:
                time.sleep(0.5)
        read(temp_service_path, remote_service_path, 
             remote_ip=remote_ip, port=port, username=username, key_filename=key_filename)
        time.sleep(0.5)
        service = np.load(temp_service_path, allow_pickle=True)
        num_grasps = service['num_grasps']
        if num_grasps == 0:
            print("no grasps detected")
        else:
            cam_joint_base = service['joint_base']
            cam_joint_direction = service['joint_direction']
            cam_affordable_position = service['affordable_position']
            joint_type = service['joint_type']
            joint_re = service['joint_re']
            grasp_score = service['grasp_score']
            grasp_width = service['grasp_width']
            grasp_depth = service['grasp_depth']
            grasp_affordance = service['grasp_affordance']
            cam_grasp_translation = service['grasp_translation']
            cam_grasp_rotation = service['grasp_rotation']
            cam_grasp_pose = np.eye(4)
            cam_grasp_pose[:3, 3] = cam_grasp_translation
            cam_grasp_pose[:3, :3] = cam_grasp_rotation
            base_joint_base = transform_pc(cam_joint_base[None, :], cam2robot)[0]
            base_joint_direction = transform_dir(cam_joint_direction[None, :], cam2robot)[0]
            base_affordable_position = transform_pc(cam_affordable_position[None, :], cam2robot)[0]
            base_grasp_pose = cam2robot @ cam_grasp_pose
            base_grasp_pose[:3, 3] += (grasp_depth - 0.05) * base_grasp_pose[:3, 0]             # TODO: hardcode to avoid collision
            if joint_type == 0:
                # TODO: only for horizontal grasp to avoid singular robot state
                flip = np.arccos(np.dot(base_grasp_pose[:3, 2], np.array([0., 0., 1.]))) / np.pi * 180.0 < 45
                if flip:
                    print("flipped")
                    base_grasp_pose[:3, 1] = -base_grasp_pose[:3, 1]
                    base_grasp_pose[:3, 2] = -base_grasp_pose[:3, 2]
                rotate = base_grasp_pose[:3, 0][2] > 0
                if rotate:
                    print("rotated")
                    target_x_axis = base_grasp_pose[:3, 0].copy()
                    target_x_axis[2] = -target_x_axis[2]
                    rotation_angle = np.arccos(np.dot(base_grasp_pose[:3, 0], target_x_axis))
                    rotation_direction = np.array([base_grasp_pose[:3, 0][0], base_grasp_pose[:3, 0][1]])
                    rotation_direction /= np.linalg.norm(rotation_direction)
                    rotation_direction = np.array([-rotation_direction[1], rotation_direction[0], 0.])
                    rotation_matrix = tf.rotation_matrix(angle=rotation_angle, direction=rotation_direction, point=base_grasp_pose[:3, 3])
                    base_grasp_pose = rotation_matrix @ base_grasp_pose
            elif joint_type == 1:
                horizontal = np.arccos(np.dot(base_grasp_pose[:3, 0], np.array([1., 0., 0.]))) / np.pi * 180.0 < 45
                if horizontal:
                    print("horizontal")
                else:
                    print("vertical")
            else:
                raise ValueError
            base_pre_grasp_pose = base_grasp_pose.copy()
            base_pre_grasp_pose[:3, 3] -= 0.05 * base_pre_grasp_pose[:3, 0]
            g2g = np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
            base_gripper_pose = np.eye(4)
            base_gripper_pose[:3, :3] = base_grasp_pose[:3, :3] @ g2g
            base_gripper_pose[:3, 3] = base_grasp_pose[:3, 3]
            base_pre_gripper_pose = np.eye(4)
            base_pre_gripper_pose[:3, :3] = base_pre_grasp_pose[:3, :3] @ g2g
            base_pre_gripper_pose[:3, 3] = base_pre_grasp_pose[:3, 3]
            np.savez("./joint.npz", joint_base=base_joint_base, joint_direction=base_joint_direction, affordable_position=base_affordable_position)
            np.savez("./grasp.npz", grasp_pose=base_grasp_pose, grasp_width=grasp_width)
        time.sleep(0.5)
        os.remove(temp_service_path)
        end_time = time.time()
        print("===> response read", end_time - start_time)
        if vis:
            if num_grasps != 0:
                visualize(base_pc_space, pc_color_space, 
                          joint_translations=base_joint_base[None, :], joint_rotations=base_joint_direction[None, :], affordable_positions=base_affordable_position[None, :], 
                          grasp_poses=base_grasp_pose[None, ...], grasp_widths=np.array([grasp_width]), grasp_depths=np.array([0.]), grasp_affordances=np.array([grasp_affordance]), 
                          whether_frame=True, whether_bbox=True, window_name="prediction")

        print("===> resetting flag")
        serviced = np.array(False)
        np.save(temp_flag_path, serviced)
        time.sleep(0.5)
        while not (os.path.isfile(temp_flag_path) and os.access(temp_flag_path, os.R_OK)):
            time.sleep(0.1)
        send(temp_flag_path, remote_flag_path, 
             remote_ip=remote_ip, port=port, username=username, key_filename=key_filename)
        time.sleep(0.5)
        os.remove(temp_flag_path)
        end_time = time.time()
        print("===> flag reset", end_time - start_time)

        print("===> starting manipulation")
        import pdb; pdb.set_trace()
        start_time = time.time()
        if num_grasps == 0:
            exit(1)
        else:
            robot.move_gripper(grasp_width)

            robot.movePose(base_pre_gripper_pose)

            robot.movePose(base_gripper_pose)
            
            # grasp
            is_graspped = robot.gripper_close()
            is_graspped = is_graspped and robot.is_grasping()
            print(is_graspped)

            # move
            real_trajectory = []
            target_trajectory = []
            wrench_trajectory = []
            robot.start_impedance_control()
            for time_step in tqdm.trange(time_steps):
                current_EE2robot = robot.readPose()
                current_wrench=  robot.readWrench()
                if joint_type == 0:
                    rotation_angle = -5.0 * task * joint_re / 180.0 * np.pi
                    delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=base_joint_direction, point=base_joint_base)
                elif joint_type == 1:
                    translation_distance = -5.0 * task / 100.0
                    delta_pose = tf.translation_matrix(base_joint_direction * translation_distance)
                else:
                    raise ValueError
                target_EE2robot = delta_pose @ current_EE2robot
                robot.movePose(target_EE2robot)
                time.sleep(0.3)
                real_trajectory.append(current_EE2robot)
                target_trajectory.append(target_EE2robot)
                wrench_trajectory.append(current_wrench)
            robot.end_impedance_control()
            real_trajectory = np.array(real_trajectory)
            target_trajectory = np.array(target_trajectory)
            wrench_trajectory = np.array(wrench_trajectory)
            np.savez("./trajectory.npz", real_trajectory=real_trajectory, target_trajectory=target_trajectory, wrench_trajectory=wrench_trajectory)
        end_time = time.time()
        print("===> manipulation done", end_time - start_time)
        robot.gripper_open()
        # robot.homing()
    except Exception as e:
        print(e)
        if camera_loaded:
            del camera
        if robot_loaded:
            del robot
