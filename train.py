import hydra
from omegaconf import DictConfig
import logging
import os
from itertools import combinations
import time
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from datasets.rconfmask_afford_point_tuple_dataset import ArticulationDataset
from models.roartnet import create_shot_encoder, create_encoder
from test import test_fn
from utilities.env_utils import setup_seed
from utilities.metrics_utils import AverageMeter, log_metrics
from utilities.data_utils import real2prob


@hydra.main(config_path='./configs', config_name='train_config', version_base='1.2')
def train(cfg:DictConfig) -> None:
    logger = logging.getLogger('train')
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    setup_seed(seed=cfg.training.seed)

    # prepare dataset
    logger.info("Preparing dataset...")
    device = cfg.training.device
    training_path = cfg.dataset.train_path
    training_categories = cfg.dataset.train_categories
    joint_num = cfg.dataset.joint_num
    resolution = cfg.dataset.resolution
    receptive_field = cfg.dataset.receptive_field
    rgb = cfg.dataset.rgb
    denoise = cfg.dataset.denoise
    normalize = cfg.dataset.normalize
    sample_points_num = cfg.dataset.sample_points_num
    sample_tuples_num = cfg.algorithm.sampling.sample_tuples_num
    tuple_more_num = cfg.algorithm.sampling.tuple_more_num
    training_dataset = ArticulationDataset(training_path, training_categories, joint_num, resolution, receptive_field, 
                                           sample_points_num, sample_tuples_num, tuple_more_num, 
                                           rgb, denoise, normalize, debug=False, vis=False, is_train=True)
    
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    training_dataloader = torch.utils.data.DataLoader(training_dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testing_training_dataset = ArticulationDataset(training_path, training_categories, joint_num, resolution, receptive_field, 
                                                   sample_points_num, sample_tuples_num, tuple_more_num, 
                                                   rgb, denoise, normalize, debug=False, vis=False, is_train=False)
    testing_training_dataloader = torch.utils.data.DataLoader(testing_training_dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testing_path = cfg.dataset.test_path
    testing_categories = cfg.dataset.test_categories
    testing_testing_dataset = ArticulationDataset(testing_path, testing_categories, joint_num, resolution, receptive_field, 
                                                  sample_points_num, sample_tuples_num, tuple_more_num, 
                                                  rgb, denoise, normalize, debug=False, vis=False, is_train=False)
    testing_testing_dataloader = torch.utils.data.DataLoader(testing_testing_dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logger.info("Prepared dataset.")
    
    # prepare model
    logger.info("Preparing model...")
    shot_hidden_dims = cfg.algorithm.shot_encoder.hidden_dims
    shot_feature_dim = cfg.algorithm.shot_encoder.feature_dim
    shot_bn = cfg.algorithm.shot_encoder.bn
    shot_ln = cfg.algorithm.shot_encoder.ln
    shot_droput = cfg.algorithm.shot_encoder.dropout
    shot_encoder = create_shot_encoder(shot_hidden_dims, shot_feature_dim, 
                                       shot_bn, shot_ln, shot_droput)
    shot_encoder = shot_encoder.cuda(device)
    overall_hidden_dims = cfg.algorithm.encoder.hidden_dims
    rot_bin_num = cfg.algorithm.voting.rot_bin_num
    overall_bn = cfg.algorithm.encoder.bn
    overall_ln = cfg.algorithm.encoder.ln
    overall_dropout = cfg.algorithm.encoder.dropout
    encoder = create_encoder(tuple_more_num, shot_feature_dim, rgb, overall_hidden_dims, rot_bin_num, joint_num, 
                             overall_bn, overall_ln, overall_dropout)
    encoder = encoder.cuda(device)
    logger.info("Prepared model.")
    
    # optimize
    logger.info("Optimizing...")
    training_start_time = time.time()
    lr = cfg.training.lr
    weight_decay = cfg.training.weight_decay
    epoch_num = cfg.training.epoch_num
    lambda_rot = cfg.training.lambda_rot
    lambda_afford = cfg.training.lambda_afford
    lambda_conf = cfg.training.lambda_conf
    voting_num = cfg.algorithm.voting.voting_num
    angle_tol = cfg.algorithm.voting.angle_tol
    translation2pc = cfg.algorithm.voting.translation2pc
    multi_candidate = cfg.algorithm.voting.multi_candidate
    candidate_threshold = cfg.algorithm.voting.candidate_threshold
    rotation_multi_neighbor = cfg.algorithm.voting.rotation_multi_neighbor
    neighbor_threshold = cfg.algorithm.voting.neighbor_threshold
    rotation_cluster = cfg.algorithm.voting.rotation_cluster
    bmm_size = cfg.algorithm.voting.bmm_size
    opt = optim.Adam([*encoder.parameters(), *shot_encoder.parameters()], lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch_num, eta_min=lr/100.0)
    scheduler_warmup = GradualWarmupScheduler(opt, multiplier=1, total_epoch=epoch_num//20, after_scheduler=scheduler)
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb'))
    iteration = 0
    for epoch in range(epoch_num):
        if epoch == 0:
            opt.zero_grad()
            opt.step()
            scheduler_warmup.step()
        
        loss_meter = AverageMeter()
        loss_tr_meter = AverageMeter()
        loss_rot_meter = AverageMeter()
        loss_afford_meter = AverageMeter()
        loss_conf_meter = AverageMeter()

        # train
        shot_encoder.train()
        encoder.train()
        logger.info("epoch: " + str(epoch) + " lr: " + str(scheduler_warmup.get_last_lr()[0]))
        tb_writer.add_scalar('lr', scheduler_warmup.get_last_lr()[0], epoch)
        with tqdm.tqdm(training_dataloader) as t:
            data_num = 0
            data_loader_start_time = time.time()
            for batch_data in t:
                if rgb:
                    pcs, pc_normals, pc_shots, pc_colors, target_trs, target_rots, target_affords, target_confs, point_idxs_all = batch_data
                    pcs, pc_normals, pc_shots, pc_colors, target_trs, target_rots, target_affords, target_confs, point_idxs_all = \
                        pcs.cuda(device), pc_normals.cuda(device), pc_shots.cuda(device), pc_colors.cuda(device), target_trs.cuda(device), target_rots.cuda(device), target_affords.cuda(device), target_confs.cuda(device), point_idxs_all.cuda(device)
                else:
                    pcs, pc_normals, pc_shots, target_trs, target_rots, target_affords, target_confs, point_idxs_all = batch_data
                    pcs, pc_normals, pc_shots, target_trs, target_rots, target_affords, target_confs, point_idxs_all = \
                        pcs.cuda(device), pc_normals.cuda(device), pc_shots.cuda(device), target_trs.cuda(device), target_rots.cuda(device), target_affords.cuda(device), target_confs.cuda(device), point_idxs_all.cuda(device)
                # (B, N, 3), (B, N, 3), (B, N, 352)(, (B, N, 3)), (B, J, N_t, 2), (B, J, N_t), (B, J, N_t, 2), (B, J, N_t), (B, N_t, 2 + N_m)
                B = pcs.shape[0]
                N = pcs.shape[1]
                J = target_trs.shape[1]
                N_t = target_trs.shape[2]
                data_num += B
                
                opt.zero_grad()
                dataloader_end_time = time.time()
                if cfg.debug:
                    logger.warning("Data loader time: " + str(dataloader_end_time - data_loader_start_time))
                
                forward_start_time = time.time()
                # shot encoder for every point
                shot_feat = shot_encoder(pc_shots)       # (B, N, N_s)
                
                # encoder for sampled point tuples
                # shot_inputs = torch.cat([shot_feat[point_idxs_all[:, i]] for i in range(0, point_idxs_all.shape[-1])], -1)        # (sample_points, feature_dim * (2 + num_more))
                # normal_inputs = torch.cat([torch.max(torch.sum(normal[point_idxs_all[:, i]] * normal[point_idxs_all[:, j]], dim=-1, keepdim=True), 
                #                                      torch.sum(-normal[point_idxs_all[:, i]] * normal[point_idxs_all[:, j]], dim=-1, keepdim=True))
                #                                      for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1)    # (sample_points, (2+num_more \choose 2))
                # coord_inputs = torch.cat([pc[point_idxs_all[:, i]] - pc[point_idxs_all[:, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], -1) # (sample_points, 3 * (2+num_more \choose 2))
                # shot_inputs = []
                # normal_inputs = []
                # coord_inputs = []
                # for b in range(pcs.shape[0]):
                #     shot_inputs.append(torch.cat([shot_feat[b][point_idxs_all[b, :, i]] for i in range(0, point_idxs_all.shape[-1])], dim=-1))    # (sample_points, feature_dim * (2 + num_more))
                #     normal_inputs.append(torch.cat([torch.max(torch.sum(normals[b][point_idxs_all[b, :, i]] * normals[b][point_idxs_all[b, :, j]], dim=-1, keepdim=True), 
                #                                      torch.sum(-normals[b][point_idxs_all[b, :, i]] * normals[b][point_idxs_all[b, :, j]], dim=-1, keepdim=True))
                #                                      for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1))   # (sample_points, (2+num_more \choose 2))
                #     coord_inputs.append(torch.cat([pcs[b][point_idxs_all[b, :, i]] - pcs[b][point_idxs_all[b, :, j]] for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1)) # (sample_points, 3 * (2+num_more \choose 2))
                # shot_inputs = torch.stack(shot_inputs, dim=0)     # (B, sample_points, feature_dim * (2 + num_more))
                # normal_inputs = torch.stack(normal_inputs, dim=0) # (B, sample_points, (2+num_more \choose 2))
                # coord_inputs = torch.stack(coord_inputs, dim=0)   # (B, sample_points, 3 * (2+num_more \choose 2))
                shot_inputs = torch.cat([
                    torch.gather(shot_feat, 1, 
                                 point_idxs_all[:, :, i:i+1].expand(
                                 (B, N_t, shot_feat.shape[-1]))) 
                    for i in range(point_idxs_all.shape[-1])], dim=-1)      # (B, N_t, N_s * (2 + N_m))
                normal_inputs = torch.cat([torch.max(
                    torch.sum(torch.gather(pc_normals, 1, 
                                           point_idxs_all[:, :, i:i+1].expand(
                                           (B, N_t, pc_normals.shape[-1]))) * 
                              torch.gather(pc_normals, 1, 
                                           point_idxs_all[:, :, j:j+1].expand(
                                           (B, N_t, pc_normals.shape[-1]))), 
                              dim=-1, keepdim=True), 
                    torch.sum(-torch.gather(pc_normals, 1, 
                                           point_idxs_all[:, :, i:i+1].expand(
                                           (B, N_t, pc_normals.shape[-1]))) * 
                              torch.gather(pc_normals, 1, 
                                           point_idxs_all[:, :, j:j+1].expand(
                                           (B, N_t, pc_normals.shape[-1]))), 
                              dim=-1, keepdim=True)) 
                    for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1)     # (B, N_t, (2+N_m \choose 2))
                coord_inputs = torch.cat([
                    torch.gather(pcs, 1, 
                                 point_idxs_all[:, :, i:i+1].expand(
                                 (B, N_t, pcs.shape[-1]))) - 
                    torch.gather(pcs, 1, 
                                 point_idxs_all[:, :, j:j+1].expand(
                                 (B, N_t, pcs.shape[-1]))) 
                    for (i, j) in combinations(np.arange(point_idxs_all.shape[-1]), 2)], dim=-1)     # (B, N_t, 3 * (2+N_m \choose 2))
                if rgb:
                    rgb_inputs = torch.cat([
                        torch.gather(pc_colors, 1, 
                                     point_idxs_all[:, :, i:i+1].expand(
                                     (B, N_t, pc_colors.shape[-1]))) 
                        for i in range(point_idxs_all.shape[-1])], dim=-1)      # (B, N_t, 3 * (2 + N_m))
                    inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs, rgb_inputs], dim=-1)
                else:
                    inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], dim=-1)
                preds = encoder(inputs)                     # (B, N_t, (2 + N_r + 2 + 1) * J)
                forward_end_time = time.time()
                if cfg.debug:
                    logger.warning("Forward time: " + str(forward_end_time - forward_start_time))

                backward_start_time = time.time()
                loss = 0
                # regression loss for translation for topk
                pred_trs = preds[:, :, 0:(2 * J)]           # (B, N_t, 2*J)
                pred_trs = pred_trs.reshape((B, N_t, J, 2)) # (B, N_t, J, 2)
                pred_trs = pred_trs.transpose(1, 2)         # (B, J, N_t, 2)
                loss_tr_ = torch.mean((pred_trs - target_trs) ** 2, dim=-1)     # (B, J, N_t)
                loss_tr_ = loss_tr_ * target_confs
                loss_tr = loss_tr_[loss_tr_ > 0]
                loss_tr = torch.mean(loss_tr)
                loss += loss_tr
                loss_tr_meter.update(loss_tr.item())
                tb_writer.add_scalar('loss/loss_tr', loss_tr.item(), iteration)

                # classification loss for rotation for topk
                pred_rots = preds[:, :, (2 * J):(-3 * J)]   # (B, N_t, rot_bin_num*J)
                pred_rots = pred_rots.reshape((B, N_t, J, rot_bin_num))     # (B, N_t, J, rot_bin_num)
                pred_rots = pred_rots.transpose(1, 2)       # (B, J, N_t, rot_bin_num)
                pred_rots_ = F.log_softmax(pred_rots, dim=-1)               # (B, J, N_t, rot_bin_num)
                target_rots_ = real2prob(target_rots, np.pi, rot_bin_num, circular=False)                   # (B, J, N_t, rot_bin_num)
                loss_rot_ = torch.sum(F.kl_div(pred_rots_, target_rots_, reduction='none'), dim=-1)         # (B, J, N_t)
                loss_rot_ = loss_rot_ * target_confs
                loss_rot = loss_rot_[loss_rot_ > 0]
                loss_rot = torch.mean(loss_rot)
                loss_rot *= lambda_rot
                loss += loss_rot
                loss_rot_meter.update(loss_rot.item())
                tb_writer.add_scalar('loss/loss_rot', loss_rot.item(), iteration)
                
                # regression loss for affordance for topk
                pred_affords = preds[:, :, (-3 * J):-J]                     # (B, N_t, 2*J)
                pred_affords = pred_affords.reshape((B, N_t, J, 2))         # (B, N_t, J, 2)
                pred_affords = pred_affords.transpose(1, 2)                 # (B, J, N_t, 2)
                loss_afford_ = torch.mean((pred_affords - target_affords) ** 2, dim=-1)     # (B, J, N_t)
                loss_afford_ = loss_afford_ * target_confs
                loss_afford = loss_afford_[loss_afford_ > 0]
                loss_afford = torch.mean(loss_afford)
                loss_afford *= lambda_afford
                loss += loss_afford
                loss_afford_meter.update(loss_afford.item())
                tb_writer.add_scalar('loss/loss_afford', loss_afford.item(), iteration)

                # classification loss for goodness
                pred_confs = preds[:, :, -J:]               # (B, N_t, J)
                pred_confs = pred_confs.transpose(1, 2)     # (B, J, N_t)
                loss_conf = F.binary_cross_entropy_with_logits(pred_confs, target_confs, reduction='none')  # (B, J, N_t)
                loss_conf = torch.mean(loss_conf)
                loss_conf *= lambda_conf
                loss += loss_conf
                loss_conf_meter.update(loss_conf.item())
                tb_writer.add_scalar('loss/loss_conf', loss_conf.item(), iteration)
                
                loss.backward(retain_graph=False)
                # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
                # torch.nn.utils.clip_grad_norm_(shot_encoder.parameters(), 1.)
                opt.step()
                backward_end_time = time.time()
                if cfg.debug:
                    logger.warning("Backward time: " + str(backward_end_time - backward_start_time))
                
                loss_meter.update(loss.item())
                tb_writer.add_scalar('loss/loss', loss.item(), iteration)
                
                t.set_postfix(epoch=epoch, loss=loss_meter.avg, tr=loss_tr_meter.avg, rot=loss_rot_meter.avg, afford=loss_afford_meter.avg, conf=loss_conf_meter.avg)
                
                iteration += 1
                data_loader_start_time = time.time()
            scheduler_warmup.step()
            tb_writer.add_scalar('loss/loss_tr_avg', loss_tr_meter.avg, epoch)
            tb_writer.add_scalar('loss/loss_rot_avg', loss_rot_meter.avg, epoch)
            tb_writer.add_scalar('loss/loss_afford_avg', loss_afford_meter.avg, epoch)
            tb_writer.add_scalar('loss/loss_conf_avg', loss_conf_meter.avg, epoch)
            tb_writer.add_scalar('loss/loss_avg', loss_meter.avg, epoch)
            logger.info("training loss: " + str(loss_tr_meter.avg) + " + " + str(loss_rot_meter.avg) + " + " + \
                        str(loss_afford_meter.avg) + " + " + str(loss_conf_meter.avg) + " = " + str(loss_meter.avg) + ", data num: " + str(data_num))
        
        # save model
        if epoch % (epoch_num // 10) == 0:
            os.makedirs(os.path.join(output_dir, 'weights'), exist_ok=True)
            torch.save(encoder.state_dict(), os.path.join(output_dir, 'weights', 'encoder_latest.pth'))
            torch.save(shot_encoder.state_dict(), os.path.join(output_dir, 'weights', 'shot_encoder_latest.pth'))

        # validation
        if cfg.training.val_training and epoch % (epoch_num // 10) == 0:
            logger.info("Validating training...")
            validating_training_start_time = time.time()
            shot_encoder.eval()
            encoder.eval()

            validating_training_num = cfg.training.val_training_num if cfg.training.val_training_num > 0 else len(testing_training_dataset)
            validating_training_results = test_fn(testing_training_dataloader, rgb, shot_encoder, encoder, 
                                                  resolution, voting_num, rot_bin_num, angle_tol, 
                                                  translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                                                  rotation_multi_neighbor, neighbor_threshold, 
                                                  bmm_size, validating_training_num, device, vis=False)
            log_metrics(validating_training_results, logger, output_dir, tb_writer, epoch, 'training')

            validating_training_end_time = time.time()
            logger.info("Validated training.")
            logger.info("Validating training time: " + str(validating_training_end_time - validating_training_start_time))

        if cfg.training.val_testing and epoch % (epoch_num // 10) == 0:
            logger.info("Validating testing...")
            validating_testing_start_time = time.time()
            shot_encoder.eval()
            encoder.eval()

            validating_testing_num = cfg.training.val_testing_num if cfg.training.val_testing_num > 0 else len(testing_testing_dataset)
            validating_testing_results = test_fn(testing_testing_dataloader, rgb, shot_encoder, encoder, 
                                                 resolution, voting_num, rot_bin_num, angle_tol, 
                                                 translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                                                 rotation_multi_neighbor, neighbor_threshold, 
                                                 bmm_size, validating_testing_num, device, vis=False)
            log_metrics(validating_testing_results, logger, output_dir, tb_writer, epoch, 'testing')

            validating_testing_end_time = time.time()
            logger.info("Validated testing.")
            logger.info("Validating testing time: " + str(validating_testing_end_time - validating_testing_start_time))
    
    training_end_time = time.time()
    logger.info("Optimized.")
    logger.info("Training time: " + str(training_end_time - training_start_time))

    # test
    if cfg.training.test_train:
        logger.info("Testing training...")
        testing_training_start_time = time.time()
        shot_encoder.eval()
        encoder.eval()

        testing_training_results = test_fn(testing_training_dataloader, rgb, shot_encoder, encoder, 
                                           resolution, voting_num, rot_bin_num, angle_tol, 
                                           translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                                           rotation_multi_neighbor, neighbor_threshold, 
                                           bmm_size, len(testing_training_dataset), device, vis=False)
        log_metrics(testing_training_results, logger, output_dir, tb_writer, epoch_num, 'training')

        testing_training_end_time = time.time()
        logger.info("Tested training.")
        logger.info("Testing training time: " + str(testing_training_end_time - testing_training_start_time))
    
    if cfg.training.test_test:
        logger.info("Testing testing...")
        testing_testing_start_time = time.time()
        shot_encoder.eval()
        encoder.eval()

        testing_testing_results = test_fn(testing_testing_dataloader, rgb, shot_encoder, encoder, 
                                          resolution, voting_num, rot_bin_num, angle_tol, 
                                          translation2pc, multi_candidate, candidate_threshold, rotation_cluster, 
                                          rotation_multi_neighbor, neighbor_threshold, 
                                          bmm_size, len(testing_testing_dataset), device, vis=False)
        log_metrics(testing_testing_results, logger, output_dir, tb_writer, epoch_num, 'testing')

        testing_testing_end_time = time.time()
        logger.info("Tested testing.")
        logger.info("Testing testing time: " + str(testing_testing_end_time - testing_testing_start_time))

    # save model
    os.makedirs(os.path.join(output_dir, 'weights'), exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(output_dir, 'weights', 'encoder_latest.pth'))
    torch.save(shot_encoder.state_dict(), os.path.join(output_dir, 'weights', 'shot_encoder_latest.pth'))


if __name__ == '__main__':
    train()
