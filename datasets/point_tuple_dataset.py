from typing import List
import os
import json
import time
import itertools
from pathlib import Path
import tqdm
import random
import numpy as np
import torch
import MinkowskiEngine as ME
import open3d as o3d

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.data_utils import pc_normalize, joints_normalize, farthest_point_sample, generate_target_tr, generate_target_rot, transform_pc, transform_dir
from utilities.vis_utils import visualize
from utilities.env_utils import setup_seed
from utilities.constants import light_blue_color, red_color, dark_red_color, dark_green_color
from src_shot.build import shot


class ArticulationDataset(torch.utils.data.Dataset):
    def __init__(self, path:str, instances:List[str], joint_num:int, resolution:float, receptive_field:int, 
                 sample_points_num:int, sample_tuples_num:int, tuple_more_num:int, 
                 rgb:bool, denoise:bool, normalize:str, debug:bool, vis:bool, is_train:bool) -> None:
        super().__init__()
        self.path = path
        self.instances = instances
        self.joint_num = joint_num
        self.resolution = resolution
        self.receptive_field = receptive_field
        self.sample_points_num = sample_points_num
        self.sample_tuples_num = sample_tuples_num
        self.tuple_more_num = tuple_more_num
        self.rgb = rgb
        self.denoise = denoise
        self.normalize = normalize
        self.debug = debug
        self.vis = vis
        self.is_train = is_train
        self.fns = sorted(list(itertools.chain(*[list(Path(path).glob('{}/*.npz'.format(instance))) for instance in instances])))
        self.permutations = list(itertools.permutations(range(self.sample_points_num), 2))
        if debug:
            print(f"{len(self.fns) =}", f"{self.fns[0] =}")

    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, idx:int):
        # load data
        data = np.load(self.fns[idx])

        pc = data['point_cloud'].astype(np.float32)                                 # (N'', 3)
        if self.debug:
            print(f"{pc.shape = }")
            print(f"{idx = }", f"{self.fns[idx] = }")
        
        if self.rgb:
            pc_color = data['rgb'].astype(np.float32)                               # (N'', 3)

        assert data['joints'].shape[0] == self.joint_num
        joints = data['joints'].astype(np.float32)                                  # (J, 9)
        if self.debug:
            print(f"{joints.shape = }")

        joint_translations = joints[:, 0:3].astype(np.float32)                      # (J, 3)
        joint_rotations = joints[:, 3:6].astype(np.float32)                         # (J, 3)
        affordable_positions = joints[:, 6:9].astype(np.float32)                    # (J, 3)
        joint_types = joints[:, -1].astype(np.int64)                                # (J,)
        if self.debug:
            print(f"{joint_translations.shape = }", f"{joint_rotations.shape = }", f"{affordable_positions.shape = }")
        
        # TODO: transform to sapien coordinate, not compatible with pybullet generated data
        transform_matrix = np.array([[0, 0, 1, 0], 
                                     [-1, 0, 0, 0], 
                                     [0, -1, 0, 0], 
                                     [0, 0, 0, 1]])
        pc = transform_pc(pc, transform_matrix)
        joint_translations = transform_pc(joint_translations, transform_matrix)
        joint_rotations = transform_dir(joint_rotations, transform_matrix)
        affordable_positions = transform_pc(affordable_positions, transform_matrix)

        if self.debug and self.vis:
            # camera coordinate as (x, y, z) = (right, down, in)
            # normal as inside/outside both
            visualize(pc, pc_color=pc_color if self.rgb else light_blue_color, pc_normal=None, 
                      joint_translations=joint_translations, joint_rotations=joint_rotations, affordable_positions=affordable_positions, 
                      joint_axis_colors=red_color, joint_point_colors=dark_red_color, affordable_position_colors=dark_green_color, 
                      whether_frame=True, whether_bbox=True, window_name='before')

        # preprocess
        suffix = 'cache'
        if self.denoise:
            suffix += '_denoise'
        suffix += f'_{self.normalize}'
        suffix += f'_{str(self.resolution)}'
        suffix += f'_{str(self.receptive_field)}'
        suffix += f'_{str(self.sample_points_num)}'
        preprocessed_path = os.path.join(os.path.dirname(str(self.fns[idx])), suffix, os.path.basename(str(self.fns[idx])))
        if os.path.exists(preprocessed_path):
            preprocessed_data = np.load(preprocessed_path)
            pc = preprocessed_data['pc'].astype(np.float32)
            pc_normal = preprocessed_data['pc_normal'].astype(np.float32)
            pc_shot = preprocessed_data['pc_shot'].astype(np.float32)
            if self.rgb:
                pc_color = preprocessed_data['pc_color'].astype(np.float32)
            center = preprocessed_data['center'].astype(np.float32)
            scale = float(preprocessed_data['scale'])
            joint_translations, joint_rotations = joints_normalize(joint_translations, joint_rotations, center, scale)
            affordable_positions, _ = joints_normalize(affordable_positions, None, center, scale)
        else:
            start_time = time.time()
            if self.denoise:
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pc)
                # _, index = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
                # pc = pc[index]
                # if self.rgb:
                #     pc_color = pc_color[index]
                valid_mask = pc[:, 0] > 0.05
                pc = pc[valid_mask]
                if self.rgb:
                    pc_color = pc_color[valid_mask]
            end_time = time.time()
            if self.debug:
                print(f"denoise: {end_time - start_time}")
                print(f"{pc.shape = }")
            
            start_time = time.time()
            pc, center, scale = pc_normalize(pc, self.normalize)
            joint_translations, joint_rotations = joints_normalize(joint_translations, joint_rotations, center, scale)
            affordable_positions, _ = joints_normalize(affordable_positions, None, center, scale)
            end_time = time.time()
            if self.debug:
                print(f"pc_normalize: {end_time - start_time}")

            start_time = time.time()
            indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=self.resolution)[1]
            pc = np.ascontiguousarray(pc[indices].astype(np.float32))                   # (N', 3)
            if self.rgb:
                pc_color = pc_color[indices]                                            # (N', 3)
            end_time = time.time()
            if self.debug:
                print(f"sparse_quantize: {end_time - start_time}")
                print(f"{pc.shape = }")

            start_time = time.time()
            pc_normal = shot.estimate_normal(pc, self.resolution * self.receptive_field).reshape(-1, 3).astype(np.float32)
            pc_normal[~np.isfinite(pc_normal)] = 0                                      # (N', 3)
            end_time = time.time()
            if self.debug:
                print(f"estimate_normal: {end_time - start_time}")
                print(f"{pc_normal.shape = }")
            
            start_time = time.time()
            pc_shot = shot.compute(pc, self.resolution * self.receptive_field, self.resolution * self.receptive_field).reshape(-1, 352).astype(np.float32)
            pc_shot[~np.isfinite(pc_shot)] = 0                                          # (N', 352)
            end_time = time.time()
            if self.debug:
                print(f"shot: {end_time - start_time}")
                print(f"{pc_shot.shape = }")

            start_time = time.time()
            pc, indices = farthest_point_sample(pc, self.sample_points_num)             # (N, 3)
            pc_normal = pc_normal[indices]                                              # (N, 3)
            pc_shot = pc_shot[indices]                                                  # (N, 352)
            if self.rgb:
                pc_color = pc_color[indices]                                            # (N, 3)
            end_time = time.time()
            if self.debug:
                print(f"farthest_point_sample: {end_time - start_time}")
                print(f"{pc.shape = }", f"{pc_normal.shape = }", f"{pc_shot.shape = }")
            
            if not self.debug:
                os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
                if self.rgb:
                    np.savez(preprocessed_path, pc=pc, pc_normal=pc_normal, pc_shot=pc_shot, pc_color=pc_color, center=center, scale=scale)
                else:
                    np.savez(preprocessed_path, pc=pc, pc_normal=pc_normal, pc_shot=pc_shot, center=center, scale=scale)
            
        if self.debug and self.vis:
            # camera coordinate as (x, y, z) = (right, down, in)
            # normal as inside/outside both
            visualize(pc, pc_color=pc_color if self.rgb else light_blue_color, pc_normal=pc_normal, 
                    joint_translations=joint_translations, joint_rotations=joint_rotations, affordable_positions=affordable_positions, 
                    joint_axis_colors=red_color, joint_point_colors=dark_red_color, affordable_position_colors=dark_green_color, 
                    whether_frame=True, whether_bbox=True, window_name='after')

        # sample point tuples
        start_time = time.time()
        point_idxs = random.sample(self.permutations, self.sample_tuples_num)
        point_idxs = np.array(point_idxs, dtype=np.int64)                           # (N_t, 2)
        point_idxs_more = np.random.randint(0, self.sample_points_num, size=(self.sample_tuples_num, self.tuple_more_num), dtype=np.int64)  # (N_t, N_m)
        point_idxs_all = np.concatenate([point_idxs, point_idxs_more], axis=-1)     # (N_t, 2 + N_m)
        end_time = time.time()
        if self.debug:
            print(f"sample_point_tuples: {end_time - start_time}")
            print(f"{point_idxs_all.shape = }")

        # generate targets
        start_time = time.time()
        targets_tr, targets_rot = [], []
        for j in range(self.joint_num):
            target_tr = generate_target_tr(pc, joint_translations[j], point_idxs_all[:, :2])    # (N_t, 2)
            target_rot = generate_target_rot(pc, joint_rotations[j], point_idxs_all[:, :2])     # (N_t,)
            targets_tr.append(target_tr)
            targets_rot.append(target_rot)
        targets_tr = np.stack(targets_tr, axis=0).astype(np.float32)                    # (J, N_t, 2)
        targets_rot = np.stack(targets_rot, axis=0).astype(np.float32)                  # (J, N_t)
        end_time = time.time()
        if self.debug:
            print(f"generate_targets: {end_time - start_time}")
            print(f"{targets_tr.shape = }", f"{targets_rot.shape = }")
    
        if self.is_train:
            if self.rgb:
                return pc, pc_normal, pc_shot, pc_color, \
                    targets_tr, targets_rot, \
                    point_idxs_all
            else:
                return pc, pc_normal, pc_shot, \
                    targets_tr, targets_rot, \
                    point_idxs_all
        else:
            if self.rgb:
                return pc, pc_normal, pc_shot, pc_color, \
                    joint_translations, joint_rotations, affordable_positions, joint_types, \
                    point_idxs_all, str(self.fns[idx])
            else:
                return pc, pc_normal, pc_shot, \
                    joint_translations, joint_rotations, affordable_positions, joint_types, \
                    point_idxs_all, str(self.fns[idx])


if __name__ == '__main__':
    setup_seed(seed=42)
    path = "/data2/junbo/RealArt-6/without_table/microwave"
    instances = ['0_without_chaos', '1_without_chaos', '2_without_chaos', '3_without_chaos', '4_without_chaos']
    joint_num = 1
    resolution = 1e-2
    receptive_field = 10
    sample_points_num = 1024
    sample_tuples_num = 100000
    tuple_more_num = 3
    normalize = 'none'
    rgb = False
    denoise = True
    dataset = ArticulationDataset(path, instances, joint_num, resolution, receptive_field, 
                                  sample_points_num, sample_tuples_num, tuple_more_num, 
                                  rgb, denoise, normalize, debug=True, vis=True, is_train=False)
    batch_size = 2
    num_workers = 0
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for results in tqdm.tqdm(dataloader):
        import pdb; pdb.set_trace()
