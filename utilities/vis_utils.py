from typing import Tuple, Union, Optional
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

from .constants import EPS


def visualize(pc:np.ndarray, pc_color:Optional[np.ndarray]=None, pc_normal:Optional[np.ndarray]=None, 
              joint_translations:Optional[np.ndarray]=None, joint_rotations:Optional[np.ndarray]=None, affordable_positions:Optional[np.ndarray]=None, 
              joint_axis_colors:Optional[np.ndarray]=None, joint_point_colors:Optional[np.ndarray]=None, affordable_position_colors:Optional[np.ndarray]=None, 
              grasp_poses:Optional[np.ndarray]=None, grasp_widths:Optional[np.ndarray]=None, grasp_depths:Optional[np.ndarray]=None, grasp_affordances:Optional[np.ndarray]=None, 
              whether_frame:bool=True, whether_bbox:Union[bool, Tuple[np.ndarray, np.ndarray]]=True, window_name:str='visualization') -> None:
    geometries = []

    if whether_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if pc_color is None:
        pass
    elif len(pc_color.shape) == 1:
        pcd.paint_uniform_color(pc_color)
    elif len(pc_color.shape) == 2:
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
    else:
        raise NotImplementedError
    if pc_normal is None:
        pass
    else:
        pcd.normals = o3d.utility.Vector3dVector(pc_normal)
    geometries.append(pcd)

    if isinstance(whether_bbox, bool) and whether_bbox:
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        bbox.color = np.array([1, 0, 0])
        geometries.append(bbox)
    elif isinstance(whether_bbox, tuple):
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=whether_bbox[0], max_bound=whether_bbox[1])
        bbox.color = np.array([1, 0, 0])
        geometries.append(bbox)
    else:
        pass

    joint_num = joint_translations.shape[0] if joint_translations is not None else 0
    for j in range(joint_num):
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(joint_rotations[j], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < EPS:
            temp1 = np.cross(np.array([0., 1., 0.]), joint_rotations[j])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(joint_rotations[j], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, joint_rotations[j])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = joint_rotations[j]
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.translate(joint_translations[j].reshape((3, 1)))
        if joint_axis_colors is None:
            pass
        elif len(joint_axis_colors.shape) == 1:
            joint_axis.paint_uniform_color(joint_axis_colors)
        elif len(joint_axis_colors.shape) == 2:
            joint_axis.paint_uniform_color(joint_axis_colors[j])
        else:
            raise NotImplementedError
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        joint_point = joint_point.translate(joint_translations[j].reshape((3, 1)))
        if joint_point_colors is None:
            pass
        elif len(joint_point_colors.shape) == 1:
            joint_point.paint_uniform_color(joint_point_colors)
        elif len(joint_point_colors.shape) == 2:
            joint_point.paint_uniform_color(joint_point_colors[j])
        else:
            raise NotImplementedError
        geometries.append(joint_point)
    
    if affordable_positions is not None:
        affordable_position_num = affordable_positions.shape[0]
        for i in range(affordable_position_num):
            affordable_position = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            affordable_position = affordable_position.translate(affordable_positions[i].reshape((3, 1)))
            if affordable_position_colors is None:
                pass
            elif len(affordable_position_colors.shape) == 1:
                affordable_position.paint_uniform_color(affordable_position_colors)
            elif len(affordable_position_colors.shape) == 2:
                affordable_position.paint_uniform_color(affordable_position_colors[i])
            else:
                raise NotImplementedError
            geometries.append(affordable_position)
    
    if grasp_poses is not None:
        grasp_num = grasp_poses.shape[0]
        for g in range(grasp_num):
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            grasp_frame.transform(grasp_poses[g])
            geometries.append(grasp_frame)

            finger_width = 0.004
            tail_length = 0.04
            depth_base = 0.02
            gg_width = grasp_widths[g]
            gg_depth = grasp_depths[g]
            gg_affordance = grasp_affordances[g]
            gg_translation = grasp_poses[g][:3, 3]
            gg_rotation = grasp_poses[g][:3, :3]

            left = np.zeros((2, 3))
            left[0] = np.array([-depth_base - finger_width, -gg_width / 2, 0])
            left[1] = np.array([gg_depth, -gg_width / 2, 0])

            right = np.zeros((2, 3))
            right[0] = np.array([-depth_base - finger_width, gg_width / 2, 0])
            right[1] = np.array([gg_depth, gg_width / 2, 0])

            bottom = np.zeros((2, 3))
            bottom[0] = np.array([-finger_width - depth_base, -gg_width / 2, 0])
            bottom[1] = np.array([-finger_width - depth_base, gg_width / 2, 0])

            tail = np.zeros((2, 3))
            tail[0] = np.array([-(tail_length + finger_width + depth_base), 0, 0])
            tail[1] = np.array([-(finger_width + depth_base), 0, 0])

            vertices = np.vstack([left, right, bottom, tail])
            vertices = np.dot(gg_rotation, vertices.T).T + gg_translation

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5], [6, 7]])
            if gg_affordance < 0.5:
                line_set.paint_uniform_color([1, 2*gg_affordance, 0])
            elif gg_affordance == 1.0:
                line_set.paint_uniform_color([0., 0., 1.])
            else:
                line_set.paint_uniform_color([-2*gg_affordance+2, 1, 0])
            geometries.append(line_set)
    
    o3d.visualization.draw_geometries(geometries, point_show_normal=pc_normal is not None, window_name=window_name)

def visualize_mask(pc:np.ndarray, instance_mask:np.ndarray, function_mask:np.ndarray, 
                   pc_normal:Optional[np.ndarray]=None, 
                   joint_translations:Optional[np.ndarray]=None, joint_rotations:Optional[np.ndarray]=None, affordable_positions:Optional[np.ndarray]=None, 
                   whether_frame:bool=True, whether_bbox:Union[bool, Tuple[np.ndarray, np.ndarray]]=True, window_name:str='visualization') -> None:
    geometries = []

    if whether_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass
    
    instance_ids = np.unique(instance_mask)     # [0, J]
    functions = []
    for instance_id in instance_ids:
        instance_function = function_mask[instance_mask == instance_id]
        functions.append(np.unique(instance_function)[0])
    revolute_num, prismatic_num, fixed_num = 0, 0, 0
    for f in functions:
        if f == 0:
            revolute_num += 1
        elif f == 1:
            prismatic_num += 1
        elif f == 2:
            fixed_num += 1
        else:
            raise ValueError(f"Invalid function {f}")
    # assert fixed_num == 1
    revolute_gradient = 1.0 / revolute_num if revolute_num > 0 else 0.0
    prismatic_gradient = 1.0 / prismatic_num if prismatic_num > 0 else 0.0

    pc_color = np.zeros((pc.shape[0], 3), dtype=np.float32)
    revolute_num, prismatic_num = 0, 0
    for instance_idx, instance_id in enumerate(instance_ids):
        if functions[instance_idx] == 0:
            pc_color[instance_mask == instance_id] = np.array([1. - revolute_gradient * revolute_num, 0., 0.])
            revolute_num += 1
        elif functions[instance_idx] == 1:
            pc_color[instance_mask == instance_id] = np.array([0., 1. - prismatic_gradient * prismatic_num, 0.])
            prismatic_num += 1
        elif functions[instance_idx] == 2:
            pc_color[instance_mask == instance_id] = np.array([0., 0., 0.])
        else:
            raise ValueError(f"Invalid function {functions[instance_idx]}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    if pc_normal is None:
        pass
    else:
        pcd.normals = o3d.utility.Vector3dVector(pc_normal)
    geometries.append(pcd)

    if isinstance(whether_bbox, bool) and whether_bbox:
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        bbox.color = np.array([1, 0, 0])
        geometries.append(bbox)
    elif isinstance(whether_bbox, tuple):
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=whether_bbox[0], max_bound=whether_bbox[1])
        bbox.color = np.array([1, 0, 0])
        geometries.append(bbox)
    else:
        pass

    joint_num = joint_translations.shape[0] if joint_translations is not None else 0
    revolute_num, prismatic_num = 0, 0
    for j in range(joint_num):
        joint_function = functions[j+1]
        if joint_function == 0:
            joint_color = np.array([1. - revolute_gradient * revolute_num, 0., 0.])
            revolute_num += 1
        elif joint_function == 1:
            joint_color = np.array([0., 1. - prismatic_gradient * prismatic_num, 0.])
            prismatic_num += 1
        else:
            raise ValueError(f"Invalid function {joint_function}")
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(joint_rotations[j], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < EPS:
            temp1 = np.cross(np.array([0., 1., 0.]), joint_rotations[j])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(joint_rotations[j], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, joint_rotations[j])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = joint_rotations[j]
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.translate(joint_translations[j].reshape((3, 1)))
        joint_axis.paint_uniform_color(joint_color)
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        joint_point = joint_point.translate(joint_translations[j].reshape((3, 1)))
        joint_point.paint_uniform_color(joint_color)
        geometries.append(joint_point)
        if affordable_positions is not None:
            affordance_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            affordance_point = affordance_point.translate(affordable_positions[j].reshape((3, 1)))
            affordance_point.paint_uniform_color(joint_color)
            geometries.append(affordance_point)
    
    o3d.visualization.draw_geometries(geometries, point_show_normal=pc_normal is not None, window_name=window_name)


def visualize_translation_voting(grid_pc:np.ndarray, votes_list:np.ndarray, 
                                 pc:Optional[np.ndarray]=None, pc_color:Optional[np.ndarray]=None, 
                                 gt_translation:Optional[np.ndarray]=None, gt_color:Optional[np.ndarray]=None, 
                                 pred_translation:Optional[np.ndarray]=None, pred_color:Optional[np.ndarray]=None, 
                                 show_threshold:float=0.5, whether_frame:bool=True, whether_bbox:bool=True, 
                                 window_name:str='visualization') -> None:
    geometries = []

    if whether_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass

    grid_pcd = o3d.geometry.PointCloud()
    votes_list = votes_list / np.max(votes_list)
    grid_pc_color = np.zeros((grid_pc.shape[0], 3))
    grid_pc_color = np.stack([np.ones_like(votes_list), 1-votes_list, 1-votes_list], axis=-1)
    grid_pcd.points = o3d.utility.Vector3dVector(grid_pc[votes_list >= show_threshold])
    grid_pcd.colors = o3d.utility.Vector3dVector(grid_pc_color[votes_list >= show_threshold])
    geometries.append(grid_pcd)
    print(grid_pc[votes_list >= show_threshold].shape[0])

    if whether_bbox:
        grid_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(grid_pc))
        grid_bbox.color = np.array([0, 1, 0])
        geometries.append(grid_bbox)
    else:
        pass

    if gt_translation is None:
        pass
    else:
        gt_point = o3d.geometry.PointCloud()
        gt_point.points = o3d.utility.Vector3dVector(gt_translation[None, :])
        if gt_color is None:
            pass
        else:
            gt_point.paint_uniform_color(gt_color)
        geometries.append(gt_point)
    
    if pred_translation is None:
        pass
    else:
        pred_point = o3d.geometry.PointCloud()
        pred_point.points = o3d.utility.Vector3dVector(pred_translation[None, :])
        if pred_color is None:
            pass
        else:
            pred_point.paint_uniform_color(pred_color)
        geometries.append(pred_point)

    if pc is None:
        pass
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if pc_color is None:
            pass
        elif len(pc_color.shape) == 1:
            pcd.paint_uniform_color(pc_color)
        elif len(pc_color.shape) == 2:
            pcd.colors = o3d.utility.Vector3dVector(pc_color)
        else:
            raise NotImplementedError
        geometries.append(pcd)

        if whether_bbox:
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
            bbox.color = np.array([1, 0, 0])
            geometries.append(bbox)
        else:
            pass
    
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def visualize_rotation_voting(sphere_pts:np.ndarray, votes:np.ndarray, 
                              pc:Optional[np.ndarray]=None, pc_color:Optional[np.ndarray]=None, 
                              gt_rotation:Optional[np.ndarray]=None, gt_color:Optional[np.ndarray]=None, 
                              pred_rotation:Optional[np.ndarray]=None, pred_color:Optional[np.ndarray]=None, 
                              show_threshold:float=0.5, whether_frame:bool=True, whether_bbox:bool=True, 
                              window_name:str='visualization') -> None:
    geometries = []

    if whether_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass

    joint_num = sphere_pts.shape[0]
    votes = votes / np.max(votes)
    print(votes[votes >= show_threshold].shape[0])
    for j in range(joint_num):
        if votes[j] < show_threshold:
            continue
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(sphere_pts[j], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < EPS:
            temp1 = np.cross(np.array([0., 1., 0.]), sphere_pts[j])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(sphere_pts[j], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, sphere_pts[j])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = sphere_pts[j]
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.paint_uniform_color(np.array([1, 1-votes[j], 1-votes[j]]))
        geometries.append(joint_axis)
    
    if gt_rotation is None:
        pass
    else:
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(gt_rotation, np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < EPS:
            temp1 = np.cross(np.array([0., 1., 0.]), gt_rotation)
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(gt_rotation, temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, gt_rotation)
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = gt_rotation
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        if gt_color is None:
            pass
        else:
            joint_axis.paint_uniform_color(gt_color)
        geometries.append(joint_axis)
    
    if pred_rotation is None:
        pass
    else:
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(pred_rotation, np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < EPS:
            temp1 = np.cross(np.array([0., 1., 0.]), pred_rotation)
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(pred_rotation, temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, pred_rotation)
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = pred_rotation
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        if pred_color is None:
            pass
        else:
            joint_axis.paint_uniform_color(pred_color)
        geometries.append(joint_axis)

    if pc is None:
        pass
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if pc_color is None:
            pass
        elif len(pc_color.shape) == 1:
            pcd.paint_uniform_color(pc_color)
        elif len(pc_color.shape) == 2:
            pcd.colors = o3d.utility.Vector3dVector(pc_color)
        else:
            raise NotImplementedError
        geometries.append(pcd)

        if whether_bbox:
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
            bbox.color = np.array([1, 0, 0])
            geometries.append(bbox)
        else:
            pass
    
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def visualize_confidence_voting(confs:np.ndarray, pc:np.ndarray, point_idxs:np.ndarray, 
                                whether_frame:bool=True, whether_bbox:bool=True, window_name:str='visualization') -> None:
    # confs: (N_t,), pc: (N, 3), point_idxs: (N_t, 2)
    point_heats = np.zeros((pc.shape[0],))                  # (N,)
    for i in range(confs.shape[0]):
        point_heats[point_idxs[i]] += confs[i]
    print(np.max(point_heats), np.min(point_heats[point_heats > 0]), point_heats[point_heats > 0].shape[0])
    point_heats /= np.max(point_heats)

    geometries = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    cmap = cm.get_cmap('jet')
    pcd.colors = o3d.utility.Vector3dVector(cmap(point_heats)[:, :3])
    geometries.append(pcd)

    if whether_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass

    if whether_bbox:
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
        bbox.color = np.array([1, 0, 0])
        geometries.append(bbox)
    else:
        pass

    o3d.visualization.draw_geometries(geometries, window_name=window_name)
