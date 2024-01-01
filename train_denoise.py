import argparse
import logging
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from transformers.modeling_bert import BertConfig
from transformers.optimization import AdamW, WarmupCosineSchedule
from tensorboardX import SummaryWriter
import numpy as np

from config_denoise import _C as config
from dataset_denoise import COCOCaptionDataset, collate_fn_train
from modeling import Generator, LabelSmoothingLoss
from utils import get_rank, mkdir, synchronize
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger
from utils.tokenizer import EOS, MASK, PAD, num_tokens
from utils.dataloader import update_sampler_weights
import measure_noise


def train_single_stage(generator, optimizer, data_loader, scheduler, checkpointer,
          device, log_time, checkpoint_time, arguments, writer, criterion):

    if arguments['config']['scheduler']['full_epochs']:
        iter_count = 0
        total_iterations = np.ceil(len(data_loader.dataset.samples['chosen']) / arguments['config']['samples_per_gpu']).astype('int')  # not tested on more the 1 gpu

    for iteration, batch in enumerate(data_loader, arguments['start_iter']):
        iteration += 1
        arguments['iteration'] = iteration

        if arguments['config']['scheduler']['full_epochs']:
            iter_count += 1

        token_type_ids = batch[0].to(device)  # (N, L), long
        input_token_ids = batch[1].to(device)  # (N, L), long
        masked_token_ids = batch[2].to(device)  # (N, L), long
        region_features = batch[3].to(device)  # (N, 100, 2048), float
        region_class = batch[4].to(device)  # (N, 100, 1601), float
        region_spatial = batch[5].to(device)  # (N, 100, 6), float

        num_img_tokens = region_spatial.size(1)
        seq_length = input_token_ids.size(1)
        batch_size = input_token_ids.size(0)

        region_spatial[:, :, [0, 2]] /= region_spatial[:, :, [2]] + 1e-5
        region_spatial[:, :, [1, 3]] /= region_spatial[:, :, [3]] + 1e-5
        rel_area = (region_spatial[:, :, [3]] - region_spatial[:, :, [1]]) * \
                   (region_spatial[:, :, [2]] - region_spatial[:, :, [0]])
        region_spatial = torch.cat((region_spatial[:, :, :4],
            rel_area.clamp_(0), region_spatial[:, :, 5:]), dim=-1)
        position_features = torch.cat((F.layer_norm(region_spatial, [6]),
            F.layer_norm(region_class, [1601])), dim=-1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_token_ids)

        region_type = position_ids.new_full(
            region_features.shape[:2], len(config.boundaries) + 1)
        token_type_ids = torch.cat((region_type, token_type_ids), dim=1)

        attention_mask = (masked_token_ids != PAD).float()
        _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        mask_position = (masked_token_ids == MASK).to(torch.long).view(-1)
        mask_position = mask_position.nonzero().squeeze()

        pred_scores = generator(
            region_features, position_features,
            masked_token_ids, token_type_ids,
            position_ids, attention_mask)

        pred_scores = pred_scores[:, num_img_tokens:, :]
        pred_scores = pred_scores.contiguous().view(-1, num_tokens)
        pred_scores = pred_scores[mask_position]

        gt_token_ids = input_token_ids.view(-1)[mask_position]
        loss = criterion(pred_scores, gt_token_ids)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(generator.parameters(), config.solver.grad_clip)
        optimizer.step()
        scheduler.step()
        batch_time = time.time() - arguments['end_time']
        arguments['end_time'] = time.time()

        if iteration % log_time == 0 or iteration == arguments['max_iter']:
            logger.info(
                '  '.join([
                    "iter: {iter}", "time: {time:.4f}", "mem: {mem:.2f}",
                    "lr: {lr:.8f}", "loss: {loss:.4f}"
                ]).format(
                    iter=iteration, time=batch_time, loss=loss,
                    lr=optimizer.param_groups[0]["lr"],
                    mem=torch.cuda.max_memory_allocated() / 1024.0 ** 3,
                ))
        if iteration % (10 * log_time) == 0:
            writer.add_scalar('training/dataset size', len(data_loader.dataset.samples['chosen']), iteration)
            writer.add_scalar('training/stage', arguments['stage'], iteration)
            writer.add_scalar('training/lr', optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar('training/loss', loss, iteration)
        if iteration % checkpoint_time == 0 or iteration == arguments['max_iter']:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        # change stage if needed
        if iteration == arguments['noisy_milestone']:  # end of noisy stage
            return iteration
        elif iteration == arguments['trusted_milestone'] + 1:  # end of trusted stage
            return iteration
        elif iteration > arguments['trusted_milestone'] and arguments['config']['scheduler']['full_epochs']:  # in case of full epochs, noisy stages
            if iter_count == total_iterations:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
                return iteration
        elif iteration > arguments['trusted_milestone'] and (iteration - arguments['trusted_milestone']) % arguments['denoise_every'] == 0:  # no full epochs, noisy stages
            return iteration


def train(generator, optimizer, data_loader, data_loader_estimation, scheduler, checkpointer,
          device, log_time, checkpoint_time, arguments, writer):
    logger = logging.getLogger("train")
    logger.info("Start training")
    max_iter = len(data_loader)
    start_iter = arguments['iteration']
    generator.train()

    if config.loss.balance_weight != 1.0:
        balance_weight = torch.ones(
            num_tokens, dtype=torch.float32, device=device)
        balance_weight[EOS] = config.loss.balance_weight
    else:
        balance_weight = None

    criterion = LabelSmoothingLoss(
        num_tokens, balance_weight, config.loss.label_smoothing)

    arguments['end_time'] = time.time()

    # set dataloader according to training stage
    if not arguments['config']['scheduler']['noisy_training']:
        training_phase = 1
        logger.info('Training on trusted data only')
    elif arguments['iteration'] < arguments['noisy_milestone']:
        training_phase = 0
    elif arguments['iteration'] <= arguments['trusted_milestone']:
        training_phase = 1
    else:
        denoising_iter = arguments['iteration'] - arguments['trusted_milestone']
        if denoising_iter % arguments['denoise_every'] == 0:
            training_phase = arguments['stage']
        else:
            training_phase = arguments['stage'] - 1  # to treat an immediate increase next (in mid stage)

    data_loader.dataset.set_train_phase(training_phase, logger)
    arguments['stage'] = training_phase
    arguments['max_iter'] = max_iter
    arguments['start_iter'] = start_iter

    # change stage if needed
    cur_iter = start_iter

    while True:
        # stop criteria
        if not arguments['config']['scheduler']['full_epochs']:
            if cur_iter >= max_iter:
                break

        if not arguments['config']['scheduler']['noisy_training']:
            arguments['start_iter'] = cur_iter
        elif cur_iter == arguments['noisy_milestone']:
            training_phase = 1
            arguments['stage'] = training_phase
            arguments['start_iter'] = cur_iter
            logger.info('Done training on noisy data')
            data_loader.dataset.set_train_phase(training_phase, logger)
        elif cur_iter == arguments['trusted_milestone'] + 1:
            if arguments['config']['do_noise_eval']:
                # call a function that measures noise (use different dataloaders)
                logger.info('Begin evaluation with noisy model')
                load_name = 'checkpoints_noisy/train/0/model_' + str(arguments['noisy_milestone']).zfill(7) + '.pth'
                measure_noise.evaluate_with_model(data_loader_estimation, load_name, 0)
                logger.info('Begin evaluation with trusted model')
                load_name = 'checkpoints_noisy/train/1/model_' + str(arguments['trusted_milestone']).zfill(7) + '.pth'
                measure_noise.evaluate_with_model(data_loader_estimation, load_name, 1)
                logger.info('Compute scores')
                measure_noise.save_scores(data_loader_estimation)
            else:
                logger.info('Skipping scores computation')
            measure_noise.load_scores(data_loader)
            training_phase = 2
            arguments['stage'] = training_phase
            arguments['start_iter'] = cur_iter
            logger.info('Done tuning on trusted data')
            data_loader.dataset.set_train_phase(training_phase, logger)
            # call a function that updates the phase & dataset itself (by sampling)
            if arguments['config']['do_weight_init']:
                # reset the model weights
                if arguments['config']['distributed']:
                    generator.module.weight_reset()
                else:
                    generator.weight_reset()
            optimizer, scheduler = set_optim(generator, arguments['config'])
            if arguments['config']['balanced_training']:
                update_sampler_weights(data_loader, data_loader.dataset, arguments['config']['distributed'], True,
                                       data_loader.dataset.boundaries)
        elif cur_iter > arguments['trusted_milestone'] + 1:
            training_phase += 1
            arguments['stage'] = training_phase
            arguments['start_iter'] = cur_iter
            if arguments['config']['scheduler']['full_epochs']:
                # max epochs to match clean training ; assumes max_iter-trusted_milestones is the #steps for clean ; not tested for single-gpu
                max_epochs = (max_iter - arguments['trusted_milestone']) * arguments['config']['samples_per_gpu'] / len(data_loader.dataset.samples['trusted'])
                max_epochs = np.ceil(max_epochs).astype('int')
                if training_phase - 1 > max_epochs:
                    logger.info('Finished %d / %d epochs' % (training_phase - 2, max_epochs))
                    break
                logger.info('Starting %d / %d epochs' % (training_phase - 1, max_epochs))

            data_loader.dataset.set_train_phase(training_phase, logger)
            if arguments['config']['balanced_training']:
                update_sampler_weights(data_loader, data_loader.dataset, arguments['config']['distributed'], True,
                                       data_loader.dataset.boundaries)
            logger.info('Sampled data for phase %d' % training_phase)

        cur_iter = train_single_stage(generator, optimizer, data_loader, scheduler, checkpointer,
          device, log_time, checkpoint_time, arguments, writer, criterion)


def set_optim(generator, config):
    optimizer = AdamW(
        params=generator.parameters(),
        lr=config.solver.lr,
        weight_decay=config.solver.weight_decay,
        betas=config.solver.betas
    )
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        t_total=config.scheduler.max_steps
    )
    return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if config.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
        synchronize()

    config.merge_from_list(args.opts)
    config.freeze()

    save_dir = os.path.join(config.save_dir, f'train')
    mkdir(save_dir)
    logger = setup_logger("train", save_dir, get_rank())
    logger.info("Running with config:\n{}".format(config))
    logger.propagate = False

    arguments = {'iteration': 0, 'noisy_milestone': config.scheduler.noisy_milestone, \
                 'trusted_milestone': config.scheduler.trusted_milestone, \
                 'denoise_every': config.scheduler.denoise_every, 'stage': 0, 'config': config}
    device = torch.device(config.device)

    bert_config = BertConfig(type_vocab_size=len(config.boundaries) + 2)
    generator = Generator(bert_config)
    generator = generator.to(device)

    optimizer, scheduler = set_optim(generator, config)

    checkpointer = Checkpointer(
        model=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        save_to_disk=get_rank() == 0,
        logger=logger
    )

    if config.model_path == '':
        generator.load_weights(config.pretrained_bert)
    else:
        extra_checkpoint_data = checkpointer.load(config.model_path)
        arguments.update(extra_checkpoint_data)
        arguments['stage'] = int(config.model_path.split('/')[-2])

    # merge configurable fields that may change from pretrained
    arguments['config']['do_noise_eval'] = config['do_noise_eval']
    arguments['config']['do_weight_init'] = config['do_weight_init']
    arguments['config']['balanced_training'] = config['balanced_training']
    arguments['config']['samples_per_gpu'] = config['samples_per_gpu']
    arguments['config']['scheduler']['noisy_training'] = config['scheduler']['noisy_training']
    arguments['config']['scheduler']['max_steps'] = config['scheduler']['max_steps']
    arguments['config']['scheduler']['denoise_every'] = config['scheduler']['denoise_every']
    arguments['config']['scheduler']['s'] = config['scheduler']['s']
    arguments['config']['scheduler']['p'] = config['scheduler']['p']
    arguments['config']['scheduler']['full_epochs'] = config['scheduler']['full_epochs']
    arguments['denoise_every'] = config['scheduler']['denoise_every']

    dataset = COCOCaptionDataset(
        root=config.data_dir,
        split='trainrestval',
        boundaries=config.boundaries,
        arguments=arguments,
    )

    dataset_est = COCOCaptionDataset(
        root=config.data_dir,
        split='trainrestval',
        boundaries=config.boundaries,
        arguments=arguments,
    )

    data_loader = make_data_loader(
        dataset=dataset,
        collate_fn=collate_fn_train,
        batch_size=config.samples_per_gpu,
        num_workers=config.num_workers,
        max_iter=config.scheduler.max_steps,
        split='trainrestval',
        is_distributed=config.distributed,
        start_iter=arguments['iteration'],
        balanced=config.balanced_training,
        boundaries=config.boundaries,
    )

    data_loader_estimation = make_data_loader(
        dataset=dataset_est,
        collate_fn=collate_fn_train,
        batch_size=config.samples_per_gpu,
        num_workers=config.num_workers,
        max_iter=None,
        split='trainrestval',
        is_distributed=config.distributed,
    )

    if config.distributed:
        generator = DistributedDataParallel(
            module=generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )

    writer = SummaryWriter()

    train(generator=generator,
          optimizer=optimizer,
          data_loader=data_loader,
          data_loader_estimation=data_loader_estimation,
          scheduler=scheduler,
          checkpointer=checkpointer,
          device=device,
          log_time=config.log_time,
          checkpoint_time=config.checkpoint_time,
          arguments=arguments,
          writer=writer)

    writer.close()
