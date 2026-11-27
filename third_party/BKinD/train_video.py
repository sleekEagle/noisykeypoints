import os
import random
import time
import warnings

import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from config.config_reader import parse_args, create_parser

from dataloader.load_dataloader import load_dataloader

from model.unsupervised_model import Model
from model.multiframe import n_input_frames  # noisykeypoints
from model.scale_sigma import multi_scale_args  # noisykeypoints
from model.warp_model import WarpModel  # noisykeypoints: our method (method: warp)

from loss.compute_loss import *
from loss.warp_loss import WarpLoss

from utils import Logger, mkdir_p, save_images
from utils.model_utils import *


best_loss = 10000

def main():

    args = parse_args(create_parser())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu
    t_start = time.time()

    # noisykeypoints: seed everything that is random before the model is built (weight init,
    # data order; DataLoader workers are seeded from torch's generator). cudnn.benchmark still
    # picks kernels non-deterministically, so two runs with one seed are close, not identical.
    # Resuming does not restore the random state.
    seed = getattr(args, 'seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print('seed:', seed)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    output_shape = (int(args.image_size/4), int(args.image_size/4))
    # noisykeypoints: `method: warp` in the config trains our keypoint-driven warping model
    # (model/warp_model.py, loss/warp_loss.py, reports/warp_method_design.md) with Adam;
    # the default `bkind` is the original model, loss and SGD, unchanged.
    args.method = getattr(args, 'method', 'bkind') or 'bkind'
    print('method:', args.method)
    if args.method == 'warp':
        model = WarpModel(args.nkpts, output_shape=output_shape)
    else:
        # noisykeypoints: `input_frames` = the keypoint network sees k frames (model/multiframe.py)
        # noisykeypoints: `multi_scale` = one decoder head per target resolution (model/scale_sigma.py)
        factors, sigma_kind = multi_scale_args(args)
        model = Model(args.nkpts, output_shape=output_shape, n_frames=n_input_frames(args),
                      n_scales=max(len(factors), 1), scale_sigma=sigma_kind)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.method == 'warp':
        loss_module = WarpLoss(args)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.5, 0.999))
    else:
        loss_module = computeLoss(args, model)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # noisykeypoints: mixed precision. GradScaler multiplies the loss before backward so small
    # float16 gradients don't round to zero; with amp off it does nothing.
    args.amp = bool(getattr(args, 'amp', False))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    print('mixed precision (amp):', args.amp)

    # optionally resume from a checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    title = 'Landmark-discovery'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, weights_only=False)  # noisykeypoints: our own checkpoints; newer torch defaults to True
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # noisykeypoints: if log.txt is missing (e.g. never uploaded to Google Drive before a
            # disconnect), start a new one instead of crashing; earlier epochs are then not in it.
            log_path = os.path.join(args.checkpoint, 'log.txt')
            if os.path.exists(log_path):
                logger = Logger(log_path, title=title, resume=True)
            else:
                print(f"=> {log_path} missing; starting a new log (epochs before {checkpoint['epoch']} not listed)")
                logger = Logger(log_path, title=title)
                logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss',])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss',])

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    valdir = os.path.join(args.data)

    train_dataset, val_dataset = load_dataloader(args)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, loss_module, 0, args)
        return

    is_best = True
    max_hours = getattr(args, 'max_hours', None)
    last_epoch_s = None
    for epoch in range(args.start_epoch, args.epochs):
        # noisykeypoints: stop between epochs if the next one would pass max_hours. The first
        # rotation epoch takes ~2x longer than the ones before it, so the estimate doubles there.
        if max_hours and last_epoch_s:
            need = last_epoch_s * (2.1 if epoch == args.curriculum else 1.1)
            if time.time() - t_start + need > max_hours * 3600:
                print(f'STOPPED EARLY: time limit ({max_hours} h). Finished epochs: {epoch}. '
                      f'Resume from {args.checkpoint}/checkpoint.pth.tar')
                return
        t_epoch = time.time()
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, model, loss_module, optimizer, epoch, args, scaler)

        # evaluate on validation set every val_schedule epochs
        if epoch > 0 and epoch%args.val_schedule == 0:
            test_loss = validate(val_loader, model, loss_module, epoch, args)
        else:
            test_loss = 10000  # set test_loss = 100000 when not using validate

        logger.append([args.lr * (0.1 ** (epoch // args.schedule)), train_loss, test_loss])

        # remember best acc@1 and save checkpoint
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'amp': args.amp,
            'seed': seed,
            'batch_size': args.batch_size,
            'method': args.method,
        }, is_best, checkpoint=args.checkpoint)
        last_epoch_s = time.time() - t_epoch
        print(f'epoch {epoch} took {last_epoch_s / 60:.1f} min', flush=True)

    print('TRAINING COMPLETE')

        
def train(train_loader, model, loss_module, optimizer, epoch, args, scaler):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    part_meters = {}  # noisykeypoints: running averages of the warp loss parts

    for i, images in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # noisykeypoints: always move inputs to the GPU. With gpu=None (multi-GPU DataParallel)
        # the original code skipped this and crashed; .cuda(None) uses the current device.
        inputs, tr_inputs, loss_mask, in_mask = images[0].cuda(args.gpu, non_blocking=True), \
                                                images[1].cuda(args.gpu, non_blocking=True), \
                                                images[2].cuda(args.gpu, non_blocking=True), \
                                                images[3].cuda(args.gpu, non_blocking=True)

        if args.method == 'warp':
            # noisykeypoints: our method. Frame A (inputs) is warped into frame B (tr_inputs); a random
            # smooth warp of B gives the equivariance targets. BKinD's rotated copies are not used.
            transform = loss_module.make_transform(tr_inputs.shape[0], tr_inputs.device)
            transformed = transform.transform_frame(tr_inputs)
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                out = model(inputs, tr_inputs, transformed)
            # images[9] = precomputed point tracks of the pair, when the config has `tracks`
            tracks = images[9].cuda(args.gpu, non_blocking=True) if len(images) > 9 else None
            loss, parts = loss_module(out, tr_inputs, transform, source=inputs,
                                      mask=loss_mask if getattr(args, 'motion_mask', None) else None,
                                      tracks=tracks)
            losses.update(loss.item(), images[0].size(0))
            for name, value in parts.items():
                part_meters.setdefault(name, AverageMeter(name, ':.4f')).update(float(value), images[0].size(0))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
                if i % (args.print_freq * 50) == 0:
                    print('   loss parts: ' + '  '.join(f'{m.name} {m.avg:.4f}' for m in part_meters.values()),
                          flush=True)
                if args.visualize:
                    kd = out['kp_driving']
                    vis = (out['prediction'], (kd['value'][..., 0], kd['value'][..., 1]), kd['heatmap'],
                           None, None, kd['confidence'])
                    # 0: BKinD's pre-curriculum layout; motion = where the loss looks (white = moved)
                    save_images(tr_inputs, vis, 0, args, epoch, motion=loss_module.last_weight)
            continue

        rot_im1, rot_im2,rot_im3  = images[4].cuda(args.gpu, non_blocking=True), \
                        images[5].cuda(args.gpu, non_blocking=True), \
                        images[6].cuda(args.gpu, non_blocking=True)

        # noisykeypoints: networks run in float16 when args.amp; losses are float32 (compute_loss.py).
        with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
            if epoch < args.curriculum:
                output = model(inputs, tr_inputs)
            else:
                output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3)
        # noisykeypoints: with `input_frames` the inputs are stacks of k frames (current frame last);
        # the loss and the change target use the current frames only.
        inputs, tr_inputs = inputs[:, -3:], tr_inputs[:, -3:]

        # noisykeypoints: images[9] = precomputed point tracks when the config has `tracks`
        tracks = images[9].cuda(args.gpu, non_blocking=True) if getattr(args, 'tracks', None) else None
        # noisykeypoints: last two items = k-frame means around A and B, when the config has `temporal`
        means = (images[-2].cuda(args.gpu, non_blocking=True), images[-1].cuda(args.gpu, non_blocking=True)) \
            if getattr(args, 'temporal', None) else None
        loss = loss_module.update_loss(inputs, tr_inputs, loss_mask, output, epoch, tracks=tracks, means=means)

        # measure accuracy and record loss
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if getattr(args, 'multi_scale', None):  # noisykeypoints: each head's loss and weight exp(-s)
                sc = loss_module.last_scale
                print('scales', ' '.join(f'x{f}: L {l:.4f} w {w:.3f}' for f, l, w in
                                         zip(sc['factors'], sc['L'].tolist(), sc['weight'].tolist())), flush=True)
            if getattr(args, 'fg_penalty', None):  # noisykeypoints: share of heatmap mass off the foreground
                print(f'fg_penalty {loss_module.last_fg_penalty:.4f}', flush=True)

            if args.visualize:
                # noisykeypoints: motion_i.png = the loss mask, or the change target when it is not BKinD's
                motion = loss_mask if getattr(args, 'motion_mask', None) else None
                if getattr(args, 'change_target', None) or getattr(args, 'multi_scale', None):
                    motion = loss_module.last_target / loss_module.last_target.amax((1, 2, 3), keepdim=True).clamp(min=1e-6)
                vis = output
                if getattr(args, 'multi_scale', None):
                    # noisykeypoints: show the head with the largest weight; print each head's loss and weight
                    j = loss_module.last_head
                    vis = (output[0][:, 3 * j:3 * j + 3],) + tuple(output[1:])
                save_images(tr_inputs, vis, epoch, args, epoch, motion=motion)

    return losses.avg

def validate(val_loader, model, loss_module, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, images in enumerate(val_loader):

            # noisykeypoints: see note in train().
            inputs, tr_inputs, loss_mask, in_mask = images[0].cuda(args.gpu, non_blocking=True), \
                                                images[1].cuda(args.gpu, non_blocking=True), \
                                                images[2].cuda(args.gpu, non_blocking=True), \
                                                images[3].cuda(args.gpu, non_blocking=True)

            if getattr(args, 'method', 'bkind') == 'warp':  # noisykeypoints: L1 of the warped frame
                with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                    out = model(inputs, tr_inputs)
                loss = (out['prediction'].float() - tr_inputs.float()).abs().mean()
                losses.update(loss.item(), images[0].size(0))
                continue

            rot_im1, rot_im2,rot_im3  = images[4].cuda(args.gpu, non_blocking=True), \
                            images[5].cuda(args.gpu, non_blocking=True), \
                            images[6].cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3)
            # Note that validate function only computes MSE loss
            loss = loss_module.criterion[1](output[0][:, :3].float() * loss_mask, tr_inputs[:, -3:] * loss_mask)

            losses.update(loss.item(), images[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg


if __name__ == '__main__':
    main()
