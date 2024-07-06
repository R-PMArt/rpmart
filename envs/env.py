"""
Modified from https://github.com/warshallrho/VAT-Mart/blob/main/code/env.py
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig, ArticulationJointType
import numpy as np
import trimesh


def process_angle_limit(x):
    if np.isneginf(x):
        x = -10
    if np.isinf(x):
        x = 10
    return x

def get_random_number(l, r):
    return np.random.rand() * (r - l) + l


class ContactError(Exception):
    pass


class SVDError(Exception):
    pass


class Env(object):
    
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500, \
            object_position_offset=0.0, succ_ratio=0.1):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True
        
        self.renderer = sapien.OptifuserRenderer(config=render_config)
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0+object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, 0]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])

        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.contact_error = False

        # visual objects
        self.visual_builder = self.scene.create_actor_builder()
        self.visual_objects = dict()

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, state='closed', target_part_id=-1, target_part_idx=-1, scale=1.0):
        # NOTE: set target_part_idx only set other joints to nearly closed, will not track the target joint
        loader = self.scene.create_urdf_loader()
        loader.scale = scale
        self.object = loader.load(urdf, {"material": material})
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.all_link_names = [l.get_name() for l in self.object.get_links()]
        self.movable_link_ids = []
        self.movable_joint_idxs = []
        self.movable_link_joint_types = []
        self.movable_link_joint_names = []

        for j_idx, j in enumerate(self.object.get_joints()):
            if j.get_dof() == 1:
                if  j.type == ArticulationJointType.REVOLUTE:
                    self.movable_link_joint_types.append(0)
                if  j.type == ArticulationJointType.PRISMATIC:
                    self.movable_link_joint_types.append(1)
                self.movable_link_joint_names.append(j.get_name())
                
                self.movable_link_ids.append(j.get_child_link().get_id())
                self.movable_joint_idxs.append(j_idx)
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        # set initial qpos
        joint_angles = []
        joint_abs_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        target_part_joint_idx = -1
        joint_idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    target_part_joint_idx = len(joint_angles)
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                elif state == 'random-middle-middle':
                    if joint_idx == target_part_idx:
                        joint_angles.append(float(get_random_number(l + 0.15*(r-l), r - 0.15*(r-l))))
                    else:
                        joint_angles.append(float(get_random_number(l, l + 0.0*(r-l))))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
                joint_abs_angles.append((joint_angles[-1]-l)/(r-l))
                joint_idx += 1

        self.object.set_qpos(joint_angles)
        if target_part_id >= 0:
            return joint_angles, target_part_joint_idx, joint_abs_angles
        return joint_angles, joint_abs_angles

    def load_real_object(self, urdf, material, joint_angles=None):
        loader = self.scene.create_urdf_loader()
        self.object = loader.load(urdf, {"material": material})
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.movable_link_ids = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                self.movable_link_ids.append(j.get_child_link().get_id())
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        if joint_angles is not None:
            self.object.set_qpos(joint_angles)

        return None

    def update_and_set_joint_angles_all(self, state='closed'):
        joint_angles = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
        self.object.set_qpos(joint_angles)
        return joint_angles

    def get_target_part_axes_new(self, target_part_id):
        joint_axes = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    mat = pos.to_transformation_matrix()
                    joint_axes = [float(-mat[0, 0]), float(-mat[1, 0]), float(mat[2, 0])]
        if joint_axes is None:
            raise ValueError('joint axes error!')

        return joint_axes

    def get_target_part_axes_dir_new(self, target_part_id):
        joint_axes = self.get_target_part_axes_new(target_part_id=target_part_id)
        axes_dir = -1
        for idx_axes_dim in range(3):
            if abs(joint_axes[idx_axes_dim]) > 0.1:
                axes_dir = idx_axes_dim
        return axes_dir

    def get_target_part_origins_new(self, target_part_id):
        joint_origins = None
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == target_part_id:
                    pos = j.get_global_pose()
                    joint_origins = pos.p.tolist()
        if joint_origins is None:
            raise ValueError('joint origins error!')

        return joint_origins

    def update_joint_angle(self, joint_angles, target_part_joint_idx, state, task_lower, push=True, pull=False, drawer=False):
        if push:
            if drawer:
                l = max(self.joint_angles_lower[target_part_joint_idx], self.joint_angles_lower[target_part_joint_idx] + task_lower)
                r = self.joint_angles_upper[target_part_joint_idx]
            else:
                l = max(self.joint_angles_lower[target_part_joint_idx], self.joint_angles_lower[target_part_joint_idx] + task_lower * np.pi / 180)
                r = self.joint_angles_upper[target_part_joint_idx]
        if pull:
            if drawer:
                l = self.joint_angles_lower[target_part_joint_idx]
                r = self.joint_angles_upper[target_part_joint_idx] - task_lower
            else:
                l = self.joint_angles_lower[target_part_joint_idx]
                r = self.joint_angles_upper[target_part_joint_idx] - task_lower * np.pi / 180
        if state == 'closed':
            joint_angles[target_part_joint_idx] = (float(l))
        elif state == 'open':
            joint_angles[target_part_joint_idx] = float(r)
        elif state == 'random-middle':
            joint_angles[target_part_joint_idx] = float(get_random_number(l, r))
        elif state == 'random-middle-open':
            joint_angles[target_part_joint_idx] = float(get_random_number(r * 0.8, r))
        elif state == 'random-closed-middle':
            if np.random.random() < 0.5:
                joint_angles[target_part_joint_idx] = float(get_random_number(l, r))
            else:
                joint_angles[target_part_joint_idx] = float(l)
        else:
            raise ValueError('ERROR: object init state %s unknown!' % state)
        return joint_angles

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()
        
        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                    self.target_object_part_joint_type = j.type
                idx += 1

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_state(self):
        qpos = self.object.get_qpos()
        return float(qpos[self.target_object_part_joint_id]) - self.joint_angles_lower[self.target_object_part_joint_id]
    
    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = True
        self.check_contact_strict = strict
        self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids
        self.contact_error = False

    def end_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict):
        self.check_contact = False
        self.check_contact_strict = strict
        self.first_timestep_check_contact = False
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self):
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid():
                raise ContactError()

    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self):
        self.contacts = self.scene.get_contacts()
        contact = False; valid = False;
        for c in self.contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            has_impulse = False
            for p in c.points:
                if abs(p.impulse @ p.impulse) > 1e-4:
                    has_impulse = True
                    break
            if has_impulse:
                if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                       contact, valid = True, True
                if (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                   (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                    if self.check_contact_strict:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                if (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                    if self.check_contact_strict:
                        self.contact_error = True
                        return False
                    else:
                        contact, valid = True, True
                # starting pose should have no collision at all
                if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
                    aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.first_timestep_check_contact:
                        self.contact_error = True
                        return False

        self.first_timestep_check_contact = False
        if contact and valid:
            self.check_contact = False
        return True
    
    def check_contact_right(self):
        contacts = self.scene.get_contacts()
        if len(contacts) < len(self.robot_gripper_actor_ids):
            # print("no enough contacts")
            return False
        # for c in contacts:
        #     aid1 = c.actor1.get_id()
        #     aid2 = c.actor2.get_id()
        #     if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
        #         (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
        #         print("right")
        #         pass
        #     elif (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
        #         (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
        #         print("unright")
        #         right = False
        #     elif (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
        #         print("hand contact")
        #         right = False
        #     elif (aid1 in self.robot_gripper_actor_ids and aid2 in self.robot_gripper_actor_ids):
        #         print("also non-successful grasp")
        #         right = False
        #     else:
        #         print(c.actor1.get_name(), c.actor2.get_name())
        right_aids = []
        for c in contacts:
            aid1 = c.actor1.get_id()
            aid2 = c.actor2.get_id()
            if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id):
                right_aids.append(aid1)
            elif (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id):
                right_aids.append(aid2)
            else:
                pass
        right = (set(right_aids) == set(self.robot_gripper_actor_ids))
        return right

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None

    def get_global_mesh(self, obj):
        final_vs = [];
        final_fs = [];
        vid = 0;
        for l in obj.get_links():
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0];
                v[:, 1] *= vscale[1];
                v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs

    def get_part_mesh(self, obj, part_id):
        final_vs = [];
        final_fs = [];
        vid = 0;
        for l in obj.get_links():
            l_id = l.get_id()
            if l_id != part_id:
                continue
            vs = []
            for s in l.get_collision_shapes():
                v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
                f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
                vscale = s.convex_mesh_geometry.scale
                v[:, 0] *= vscale[0];
                v[:, 1] *= vscale[1];
                v[:, 2] *= vscale[2];
                ones = np.ones((v.shape[0], 1), dtype=np.float32)
                v_ones = np.concatenate([v, ones], axis=1)
                transmat = s.pose.to_transformation_matrix()
                v = (v_ones @ transmat.T)[:, :3]
                vs.append(v)
                final_fs.append(f + vid)
                vid += v.shape[0]
            if len(vs) > 0:
                vs = np.concatenate(vs, axis=0)
                ones = np.ones((vs.shape[0], 1), dtype=np.float32)
                vs_ones = np.concatenate([vs, ones], axis=1)
                transmat = l.get_pose().to_transformation_matrix()
                vs = (vs_ones @ transmat.T)[:, :3]
                final_vs.append(vs)
        final_vs = np.concatenate(final_vs, axis=0)
        final_fs = np.concatenate(final_fs, axis=0)
        return final_vs, final_fs

    def sample_pc(self, v, f, n_points=4096):
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
        return points

    def check_drawer(self):
        for j in self.object.get_joints():
            if j.get_dof() == 1 and (j.type == ArticulationJointType.PRISMATIC):
                return True
        return False
    
    def add_point_visual(self, point:np.ndarray, color=[1, 0, 0], radius=0.04, name='point_visual'):
        self.visual_builder.add_sphere_visual(pose=Pose(p=point), radius=radius, color=color, name=name)
        point_visual = self.visual_builder.build_static(name=name)
        assert name not in self.visual_objects.keys()
        self.visual_objects[name] = point_visual
    
    def add_line_visual(self, point1:np.ndarray, point2:np.ndarray, color=[1, 0, 0], width=0.03, name='line_visual'):
        direction = point2 - point1
        direction = direction / np.linalg.norm(direction)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(direction, np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < 1e-6:
            temp1 = np.cross(np.array([0., 1., 0.]), direction)
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(direction, temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, direction)
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = direction
        pose_transformation = np.eye(4)
        pose_transformation[:3, 3] = (point1+point2)/2
        pose_transformation[:3, :3] = rotation
        pose = Pose().from_transformation_matrix(pose_transformation)
        size = [width/2, width/2, np.linalg.norm(point1 - point2)/2]
        self.visual_builder.add_box_visual(pose=pose, size=size, color=color, name=name)
        line_visual = self.visual_builder.build_static(name=name)
        assert name not in self.visual_objects.keys()
        self.visual_objects[name] = line_visual
    
    def add_grasp_visual(self, grasp_width, grasp_depth, grasp_translation, grasp_rotation, affordance=0.5, name='grasp_visual'):
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        gg_width = grasp_width
        gg_depth = grasp_depth
        gg_translation = grasp_translation
        gg_rotation = grasp_rotation

        left = np.zeros((2, 3))
        left[0] = np.array([-depth_base - finger_width, -gg_width / 2, 0])
        left[1] = np.array([gg_depth, -gg_width / 2, 0])

        right = np.zeros((2, 3))
        right[0] = np.array([-depth_base - finger_width, gg_width / 2, 0])
        right[1] = np.array([gg_depth, gg_width / 2, 0])

        bottom = np.zeros((2, 3))
        bottom[0] = np.array([-depth_base - finger_width, -gg_width / 2, 0])
        bottom[1] = np.array([-depth_base - finger_width, gg_width / 2, 0])

        tail = np.zeros((2, 3))
        tail[0] = np.array([-(tail_length + finger_width + depth_base), 0, 0])
        tail[1] = np.array([-(finger_width + depth_base), 0, 0])

        vertices = np.vstack([left, right, bottom, tail])
        vertices = np.dot(gg_rotation, vertices.T).T + gg_translation

        if affordance < 0.5:
            color = [1, 2*affordance, 0]
        elif affordance == 1.0:
            color = [0, 0, 1]
        else:
            color = [-2*affordance+2, 1, 0]
        self.add_line_visual(vertices[0], vertices[1], color, width=0.005, name=name+'_left')
        self.add_line_visual(vertices[2], vertices[3], color, width=0.005, name=name+'_right')
        self.add_line_visual(vertices[4], vertices[5], color, width=0.005, name=name+'_bottom')
        self.add_line_visual(vertices[6], vertices[7], color, width=0.005, name=name+'_tail')
    
    def add_frame_visual(self):
        self.add_line_visual(np.array([0, 0, 0]), np.array([1, 0, 0]), [1, 0, 0], width=0.005, name='frame_x')
        self.add_line_visual(np.array([0, 0, 0]), np.array([0, 1, 0]), [0, 1, 0], width=0.005, name='frame_y')
        self.add_line_visual(np.array([0, 0, 0]), np.array([0, 0, 1]), [0, 0, 1], width=0.005, name='frame_z')
    
    def remove_visual(self, name):
        self.visual_builder.remove_visual_at(self.visual_objects[name].get_id())
        self.scene.remove_actor(self.visual_objects[name])
        del self.visual_objects[name]
    
    def remove_grasp_visual(self, name):
        self.remove_visual(name+'_left')
        self.remove_visual(name+'_right')
        self.remove_visual(name+'_bottom')
        self.remove_visual(name+'_tail')
    
    def remove_frame_visual(self):
        self.remove_visual('frame_x')
        self.remove_visual('frame_y')
        self.remove_visual('frame_z')
    
    def remove_all_visuals(self):
        names = list(self.visual_objects.keys())
        for name in names:
            self.remove_visual(name)
