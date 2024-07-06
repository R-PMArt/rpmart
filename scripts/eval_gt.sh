export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python env_eval.py --num_config_per_object 100 --data_path /data2/junbo/where2act_modified_sapien_dataset/7119/mobility_vhacd.urdf --cat Microwave --gt_path /data2/junbo/where2act_modified_sapien_dataset/7119/joint_abs_pose.json --graspnet --grasp --selected_part 0 --task pull --abbr 7119 --seed 42

export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python env_eval.py --num_config_per_object 100 --data_path /data2/junbo/where2act_modified_sapien_dataset/7119/mobility_vhacd.urdf --cat Microwave --gt_path /data2/junbo/where2act_modified_sapien_dataset/7119/joint_abs_pose.json --graspnet --grasp --selected_part 0 --task push --abbr 7119 --seed 43

export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python env_eval.py --num_config_per_object 100 --data_path /data2/junbo/where2act_modified_sapien_dataset/7263/mobility_vhacd.urdf --cat Microwave --gt_path /data2/junbo/where2act_modified_sapien_dataset/7263/joint_abs_pose.json --graspnet --grasp --selected_part 0 --task pull --abbr 7263 --seed 44

export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python env_eval.py --num_config_per_object 100 --data_path /data2/junbo/where2act_modified_sapien_dataset/7263/mobility_vhacd.urdf --cat Microwave --gt_path /data2/junbo/where2act_modified_sapien_dataset/7263/joint_abs_pose.json --graspnet --grasp --selected_part 0 --task push --abbr 7263 --seed 45

export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python env_eval.py --num_config_per_object 100 --data_path /data2/junbo/where2act_modified_sapien_dataset/7296/mobility_vhacd.urdf --cat Microwave --gt_path /data2/junbo/where2act_modified_sapien_dataset/7296/joint_abs_pose.json --graspnet --grasp --selected_part 0 --task pull --abbr 7296 --seed 46

export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python env_eval.py --num_config_per_object 100 --data_path /data2/junbo/where2act_modified_sapien_dataset/7296/mobility_vhacd.urdf --cat Microwave --gt_path /data2/junbo/where2act_modified_sapien_dataset/7296/joint_abs_pose.json --graspnet --grasp --selected_part 0 --task push --abbr 7296 --seed 47
