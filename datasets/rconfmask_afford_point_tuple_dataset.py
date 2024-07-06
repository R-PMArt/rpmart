from typing import List
import os
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
from utilities.data_utils import transform_pc, transform_dir, pc_normalize, joints_normalize, farthest_point_sample, generate_target_tr, generate_target_rot, pc_noise
from utilities.vis_utils import visualize_mask, visualize_confidence_voting
from utilities.env_utils import setup_seed
from src_shot.build import shot


class ArticulationDataset(torch.utils.data.Dataset):
    def __init__(self, path:str, categories:List[str], joint_num:int, resolution:float, receptive_field:int, 
                 sample_points_num:int, sample_tuples_num:int, tuple_more_num:int, 
                 noise:bool, distortion_rate:float, distortion_level:float, outlier_rate:float, outlier_level:float, 
                 rgb:bool, denoise:bool, normalize:str, debug:bool, vis:bool, is_train:bool) -> None:
        super().__init__()
        self.path = path
        self.categories = categories
        self.joint_num = joint_num
        self.resolution = resolution
        self.receptive_field = receptive_field
        self.sample_points_num = sample_points_num
        self.sample_tuples_num = sample_tuples_num
        self.tuple_more_num = tuple_more_num
        self.noise = noise                              # NOTE: only used in testing
        self.distortion_rate = distortion_rate
        self.distortion_level = distortion_level
        self.outlier_rate = outlier_rate
        self.outlier_level = outlier_level
        self.rgb = rgb
        self.denoise = denoise
        self.normalize = normalize
        self.debug = debug
        self.vis = vis
        self.is_train = is_train
        self.fns = sorted(list(itertools.chain(*[list(Path(path).glob('{}*/*.npz'.format(category))) for category in categories])))
        self.permutations = list(itertools.permutations(range(self.sample_points_num), 2))
        if debug:
            print(f"{len(self.fns) =}", f"{self.fns[0] =}")

    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, idx:int):
        # load data
        data = np.load(self.fns[idx])

        c2w = data['extrinsic'].astype(np.float32)                                  # (4, 4)
        w2c = np.linalg.inv(c2w)
        pc = data['pcd_world'].astype(np.float32)                                   # (N'', 3)
        pc = transform_pc(pc, w2c)
        if self.noise:
            pc = pc_noise(pc, self.distortion_rate, self.distortion_level, self.outlier_rate, self.outlier_level)
        if self.debug:
            print(f"{pc.shape = }")
            print(f"{idx = }", f"{self.fns[idx] = }")
        
        if self.rgb:
            pc_color = data['pcd_color'].astype(np.float32)                         # (N'', 3)
        
        instance_mask = data['instance_mask'].astype(np.int64)                      # (N'',)
        function_mask = data['function_mask'].astype(np.int64)                      # (N'',)

        joint_translations = data['joint_bases'].astype(np.float32)                 # (J, 3)
        joint_translations = transform_pc(joint_translations, w2c)
        joint_rotations = data['joint_directions'].astype(np.float32)               # (J, 3)
        joint_rotations = transform_dir(joint_rotations, w2c)
        affordable_positions = data['affordable_positions'].astype(np.float32)      # (J, 3)
        affordable_positions = transform_pc(affordable_positions, w2c)
        joint_num = joint_translations.shape[0]
        assert self.joint_num == joint_num
        if self.debug:
            print(f"{joint_translations.shape = }", f"{joint_rotations.shape = }", f"{affordable_positions.shape = }")
        
        if self.debug and self.vis:
            # camera coordinate as (x, y, z) = (in, left, up)
            # normal as inside/outside both
            visualize_mask(pc, instance_mask, function_mask, pc_normal=None, 
                           joint_translations=joint_translations, joint_rotations=joint_rotations, affordable_positions=affordable_positions, 
                           whether_frame=True, whether_bbox=True, window_name='before')

        # preprocess
        start_time = time.time()
        if self.denoise:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            _, index = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
            pc = pc[index]
            if self.rgb:
                pc_color = pc_color[index]
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
        instance_mask = instance_mask[indices]                                      # (N',)
        function_mask = function_mask[indices]                                      # (N',)
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
        instance_mask = instance_mask[indices]                                      # (N,)
        function_mask = function_mask[indices]                                      # (N,)
        end_time = time.time()
        if self.debug:
            print(f"farthest_point_sample: {end_time - start_time}")
            print(f"{pc.shape = }", f"{pc_normal.shape = }", f"{pc_shot.shape = }")
        
        if self.debug and self.vis:
            # camera coordinate as (x, y, z) = (in, left, up)
            # normal as inside/outside both
            visualize_mask(pc, instance_mask, function_mask, pc_normal=pc_normal, 
                           joint_translations=joint_translations, joint_rotations=joint_rotations, affordable_positions=affordable_positions, 
                           whether_frame=True, whether_bbox=True, window_name='after')

        # sample point tuples
        start_time = time.time()
        point_idxs = random.sample(self.permutations, self.sample_tuples_num)
        point_idxs = np.array(point_idxs, dtype=np.int64)                           # (N_t, 2)
        point_idxs_more = np.random.randint(0, self.sample_points_num, size=(self.sample_tuples_num, self.tuple_more_num), dtype=np.int64)  # (N_t, N_m)
        point_idxs_all = np.concatenate([point_idxs, point_idxs_more], axis=-1)     # (N_t, 2 + N_m)
        base_mask = (instance_mask == 0)
        base_idxs = np.where(base_mask)[0]
        part_idxs = []
        for j in range(joint_num):
            part_mask = (instance_mask == j + 1)
            part_idxs.append(np.where(part_mask)[0])
        base_idxs_mask = np.isin(point_idxs, base_idxs)
        base_case_mask = np.all(base_idxs_mask, axis=-1)
        part_base_case_mask = np.logical_and(np.any(base_idxs_mask, axis=-1), np.logical_not(base_case_mask))
        part_case_mask = np.logical_not(np.logical_or(base_case_mask, part_base_case_mask))
        end_time = time.time()
        if self.debug:
            print(f"sample_point_tuples: {end_time - start_time}")
            print(f"{point_idxs.shape = }, {point_idxs_all.shape = }")
        if self.debug and self.vis:
            visualize_confidence_voting(np.ones((self.sample_tuples_num,)), pc, point_idxs, whether_frame=True, whether_bbox=True, window_name='point tuples')

        # generate targets
        start_time = time.time()
        targets_tr = np.zeros((joint_num, self.sample_tuples_num, 2), dtype=np.float32)         # (J, N_t, 2)
        targets_rot = np.zeros((joint_num, self.sample_tuples_num), dtype=np.float32)           # (J, N_t)
        targets_afford = np.zeros((joint_num, self.sample_tuples_num, 2), dtype=np.float32)     # (J, N_t, 2)
        targets_conf = np.zeros((joint_num, self.sample_tuples_num), dtype=np.float32)          # (J, N_t)
        for j in range(joint_num):
            this_part_mask = np.any(np.isin(point_idxs[part_base_case_mask], part_idxs[j]), axis=-1)
            same_part_mask = np.all(np.isin(point_idxs[part_case_mask], part_idxs[j]), axis=-1)
            merge_this_part_mask = part_base_case_mask.copy()
            merge_this_part_mask[part_base_case_mask] = this_part_mask.copy()
            merge_same_part_mask = part_case_mask.copy()
            merge_same_part_mask[part_case_mask] = same_part_mask.copy()
            targets_tr[j] = generate_target_tr(pc, joint_translations[j], point_idxs)
            targets_rot[j] = generate_target_rot(pc, joint_rotations[j], point_idxs)
            targets_afford[j] = generate_target_tr(pc, affordable_positions[j], point_idxs)
            # targets_conf[j, merge_same_part_mask] = 0.51
            targets_conf[j, merge_this_part_mask] = 1.0
        end_time = time.time()
        if self.debug:
            print(f"generate_targets: {end_time - start_time}")
            print(f"{targets_tr.shape = }", f"{targets_rot.shape = }", f"{targets_conf.shape = }", f"{targets_afford.shape = }")
    
        if self.is_train:
            if self.rgb:
                return pc, pc_normal, pc_shot, pc_color, \
                    targets_tr, targets_rot, targets_afford, targets_conf, \
                    point_idxs_all
            else:
                return pc, pc_normal, pc_shot, \
                    targets_tr, targets_rot, targets_afford, targets_conf, \
                    point_idxs_all
        else:
            # actually during testing, the targets should not be known, here they are used to test gt
            if self.rgb:
                return pc, pc_normal, pc_shot, pc_color, \
                    joint_translations, joint_rotations, affordable_positions, \
                    targets_tr, targets_rot, targets_afford, targets_conf, \
                    point_idxs_all
            else:
                return pc, pc_normal, pc_shot, \
                    joint_translations, joint_rotations, affordable_positions, \
                    targets_tr, targets_rot, targets_afford, targets_conf, \
                    point_idxs_all


if __name__ == '__main__':
    setup_seed(seed=42)
    path = "/data2/junbo/sapien4/test"
    categories = ['Microwave']
    joint_num = 1
    # resolution = 2.5e-2
    resolution = 1e-2
    receptive_field = 10
    sample_points_num = 1024
    sample_tuples_num = 100000
    tuple_more_num = 3
    # normalize = 'median'
    normalize = 'none'
    noise = True
    distortion_rate = 0.1
    distortion_level = 0.01
    outlier_rate = 0.001
    outlier_level = 1.0
    rgb = True
    denoise = False
    dataset = ArticulationDataset(path, categories, joint_num, resolution, receptive_field, 
                                  sample_points_num, sample_tuples_num, tuple_more_num, 
                                  noise, distortion_rate, distortion_level, outlier_rate, outlier_level, 
                                  rgb, denoise, normalize, debug=True, vis=False, is_train=True)
    batch_size = 1
    num_workers = 0
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for results in tqdm.tqdm(dataloader):
        import pdb; pdb.set_trace()
