export OMP_NUM_THREADS=16; CUDA_VISIBLE_DEVICES=0 python real_service.py --cat Microwave --roartnet --roartnet_config_path configs/eval_config.yaml --graspnet --selected_part 0 --seed 42
