"""
Modified from https://github.com/warshallrho/VAT-Mart/blob/main/code/robots/panda_robot.py
    Franka Panda Robot Arm
        support panda.urdf, panda_gripper.urdf
"""

from __future__ import division
import numpy as np
from PIL import Image
from sapien.core import Pose

from .env import Env


def rot2so3(rotation):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        raise RuntimeError
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta

def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def pose2exp_coordinate(pose):
    """
    Compute the exponential coordinate corresponding to the given SE(3) matrix
    Note: unit twist is not a unit vector

    Args:
        pose: (4, 4) transformation matrix

    Returns:
        Unit twist: (6, ) vector represent the unit twist
        Theta: scalar represent the quantity of exponential coordinate
    """
    omega, theta = rot2so3(pose[:3, :3])
    ss = skew(omega)
    inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * ss + (
            1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
    v = inv_left_jacobian @ pose[:3, 3]
    return np.concatenate([omega, v]), theta

def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint


class Robot(object):
    def __init__(self, env:Env, urdf, material, open_gripper=False, scale=1.0):
        self.env = env
        self.timestep = env.scene.get_timestep()

        # load robot
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = scale
        self.robot = loader.load(urdf, {"material": material})
        self.robot.name = "robot"
        self.max_gripper_width = 0.08
        self.tcp2ee_length = 0.11
        self.scale = scale

        # hand (EE), two grippers, the rest arm joints (if any)
        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'panda_hand'][0]
        self.root2ee = self.end_effector.get_pose().to_transformation_matrix() @ self.robot.get_root_pose().inv().to_transformation_matrix()
        self.hand_actor_id = self.end_effector.get_id()
        self.gripper_joints = [joint for joint in self.robot.get_joints() if 
                joint.get_name().startswith("panda_finger_joint")]
        self.gripper_actor_ids = [joint.get_child_link().get_id() for joint in self.gripper_joints]
        self.arm_joints = [joint for joint in self.robot.get_joints() if
                joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]
        self.g2g = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])

        # set drive joint property
        for joint in self.arm_joints:
            joint.set_drive_property(1000, 400)
        for joint in self.gripper_joints:
            joint.set_drive_property(200, 60)

        # open/close the gripper at start
        if open_gripper:
            joint_angles = []
            for j in self.robot.get_joints():
                if j.get_dof() == 1:
                    if j.get_name().startswith("panda_finger_joint"):
                        joint_angles.append(self.max_gripper_width / 2.0 * scale)
                    else:
                        joint_angles.append(0)
            self.robot.set_qpos(joint_angles)
    
    def load_gripper(self, urdf, material, open_gripper=False, scale=1.0):
        raise NotImplementedError
        self.timestep = self.env.scene.get_timestep()

        # load robot
        loader = self.env.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = scale
        self.robot = loader.load(urdf, {"material": material})
        self.robot.name = "robot"
        self.max_gripper_width = 0.08
        self.tcp2ee_length = 0.11
        self.scale = scale

        # hand (EE), two grippers, the rest arm joints (if any)
        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'panda_hand'][0]
        self.root2ee = self.end_effector.get_pose().to_transformation_matrix() @ self.robot.get_root_pose().inv().to_transformation_matrix()
        self.hand_actor_id = self.end_effector.get_id()
        self.gripper_joints = [joint for joint in self.robot.get_joints() if 
                joint.get_name().startswith("panda_finger_joint")]
        self.gripper_actor_ids = [joint.get_child_link().get_id() for joint in self.gripper_joints]
        self.arm_joints = [joint for joint in self.robot.get_joints() if
                joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]

        # set drive joint property
        for joint in self.arm_joints:
            joint.set_drive_property(1000, 400)
        for joint in self.gripper_joints:
            joint.set_drive_property(200, 60)

        # open/close the gripper at start
        if open_gripper:
            joint_angles = []
            for j in self.robot.get_joints():
                if j.get_dof() == 1:
                    if j.get_name().startswith("panda_finger_joint"):
                        joint_angles.append(self.max_gripper_width / 2.0 * scale)
                    else:
                        joint_angles.append(0)
            self.robot.set_qpos(joint_angles)

    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """
        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        ee_jacobian = np.zeros([6, self.robot.dof - 2])
        ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3: self.end_effector_index * 6, :self.robot.dof - 2]
        ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6: self.end_effector_index * 6 - 3, :self.robot.dof - 2]

        inverse_jacobian = np.linalg.pinv(ee_jacobian, rcond=1e-2)
        return inverse_jacobian @ twist

    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.

        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == len(self.arm_joints)
        target_qpos = qvel * self.timestep + self.robot.get_drive_target()[:-2]
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def calculate_twist(self, time_to_target, target_ee_pose):
        relative_transform = self.end_effector.get_pose().inv().to_transformation_matrix() @ target_ee_pose
        unit_twist, theta = pose2exp_coordinate(relative_transform)
        velocity = theta / time_to_target
        body_twist = unit_twist * velocity
        current_ee_pose = self.end_effector.get_pose().to_transformation_matrix()
        return adjoint_matrix(current_ee_pose) @ body_twist
    
    def set_pose(self, target_ee_pose:np.ndarray, gripper_depth:float):
        # target_ee_pose: (4, 4) transformation of robot tcp in world frame, as grasp pose
        root_pose = np.identity(4)
        root_pose[:3, :3] = target_ee_pose[:3, :3] @ self.g2g
        root_pose[:3, 3] = target_ee_pose[:3, 3]
        root_pose[:3, 3] -= (self.tcp2ee_length * self.scale - gripper_depth) * root_pose[:3, 2]
        root_pose = np.linalg.inv(self.root2ee) @ root_pose
        self.robot.set_root_pose(Pose().from_transformation_matrix(root_pose))
    
    def get_pose(self, gripper_depth:float) -> np.ndarray:
        target_ee_pose = np.identity(4)
        # root_pose = self.robot.get_root_pose().to_transformation_matrix()
        root_pose = self.end_effector.get_pose().to_transformation_matrix()
        target_ee_pose[:3, 3] = root_pose[:3, 3] + (self.tcp2ee_length * self.scale - gripper_depth) * root_pose[:3, 2]
        target_ee_pose[:3, :3] = root_pose[:3, :3] @ np.linalg.inv(self.g2g)
        return target_ee_pose

    def move_to_target_pose(self, target_ee_pose: np.ndarray, gripper_depth:float, num_steps: int, visu=None, vis_gif=False, vis_gif_interval=200, cam=None) -> None:
        """
        Move the robot hand dynamically to a given target pose
        Args:
            target_ee_pose: (4, 4) transformation of robot tcp in world frame, as grasp pose
            num_steps:  how much steps to reach to target pose, 
                        each step correspond to self.scene.get_timestep() seconds
                        in physical simulation
        """
        if visu:
            waypoints = []
        if vis_gif:
            imgs = []

        executed_time = num_steps * self.timestep

        target_ee_root_pose = np.identity(4)
        target_ee_root_pose[:3, :3] = target_ee_pose[:3, :3] @ self.g2g
        target_ee_root_pose[:3, 3] = target_ee_pose[:3, 3]
        target_ee_root_pose[:3, 3] -= (self.tcp2ee_length * self.scale - gripper_depth) * target_ee_root_pose[:3, 2]

        spatial_twist = self.calculate_twist(executed_time, target_ee_root_pose)
        for i in range(num_steps):
            if i % 100 == 0:
                spatial_twist = self.calculate_twist((num_steps - i) * self.timestep, target_ee_root_pose)
            qvel = self.compute_joint_velocity_from_twist(spatial_twist)
            self.internal_controller(qvel)
            self.env.step()
            self.env.render()
            if visu and i % 200 == 0:
                waypoints.append(self.robot.get_qpos().tolist())
            if vis_gif and ((i + 1) % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)
            if vis_gif and (i == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                for idx in range(5):
                    imgs.append(fimg)

        if visu and not vis_gif:
            return waypoints
        if vis_gif and not visu:
            return imgs
        if visu and vis_gif:
            return imgs, waypoints

    def move_to_target_qvel(self, qvel) -> None:

        """
        Move the robot hand dynamically to a given target pose
        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps:  how much steps to reach to target pose, 
                        each step correspond to self.scene.get_timestep() seconds
                        in physical simulation
        """
        assert qvel.size == len(self.arm_joints)
        for idx_step in range(100):
            target_qpos = qvel * self.timestep + self.robot.get_drive_target()[:-2]
            for i, joint in enumerate(self.arm_joints):
                joint.set_drive_velocity_target(qvel[i])
                joint.set_drive_target(target_qpos[i])
            passive_force = self.robot.compute_passive_force()
            self.robot.set_qf(passive_force)
            self.env.step()
            self.env.render()
        return


    def close_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.0)

    def open_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(self.max_gripper_width / 2.0)
    
    def set_gripper(self, width:float):
        joint_angles = []
        for j in self.robot.get_joints():
            if j.get_dof() == 1:
                if j.get_name().startswith("panda_finger_joint"):
                    joint_angles.append(max(min(width, self.max_gripper_width) / 2.0, 0.0))
                else:
                    joint_angles.append(0)
        self.robot.set_qpos(joint_angles)
        for joint in self.gripper_joints:
            joint.set_drive_target(max(min(width, self.max_gripper_width) / 2.0, 0.0))
    
    def get_gripper(self) -> float:
        joint_angles = self.robot.get_qpos()
        width = 0.0
        j_idx = 0
        for j in self.robot.get_joints():
            if j.get_dof() == 1:
                if j.get_name().startswith("panda_finger_joint"):
                    width += joint_angles[j_idx]
                else:
                    pass
                j_idx += 1
        return width

    def clear_velocity_command(self):
        for joint in self.arm_joints:
            joint.set_drive_velocity_target(0)

    def wait_n_steps(self, n: int, visu=None, vis_gif=False, vis_gif_interval=200, cam=None):
        imgs = []
        if visu:
            waypoints = []
        self.clear_velocity_command()
        for i in range(n):
            passive_force = self.robot.compute_passive_force()
            self.robot.set_qf(passive_force)
            self.env.step()
            self.env.render()
            if visu and i % 200 == 0:
                waypoints.append(self.robot.get_qpos().tolist())
            if vis_gif and ((i + 1) % vis_gif_interval == 0):
                rgb_pose, _ = cam.get_observation()
                fimg = (rgb_pose*255).astype(np.uint8)
                fimg = Image.fromarray(fimg)
                imgs.append(fimg)
            #
        self.robot.set_qf([0] * self.robot.dof)
        if visu and vis_gif:
            return imgs, waypoints

        if visu:
            return waypoints
        if vis_gif:
            return imgs

