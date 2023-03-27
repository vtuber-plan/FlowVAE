import os
# import json
# import argparse
# import itertools
# import math
import torch
# from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torch.multiprocessing as mp
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
# from torch.utils.data.distributed import DistributedSampler

import commons
import utils
# from hubert_data import (
#     AudioSpeakerHubertLoader,
#     AudioSpeakerUnitCollate
# )
from dataset import (
    AudioSpeakerLoader,
    AudioSpeakerCollate
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    constractive_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True
global_step = 0
which_device = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    hps = utils.get_hparams()
    # 临时改成单卡训练
    run(which_device, hps)


def run(rank, hps):
    global global_step
    if rank == which_device:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(
            log_dir=os.path.join(hps.model_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = AudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = AudioSpeakerCollate(hps.data)
    train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size, num_workers=8,
                              shuffle=False, pin_memory=True, collate_fn=collate_fn)
    if rank == which_device:
        eval_dataset = AudioSpeakerLoader(
            hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
                                 batch_size=hps.train.batch_size, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    # net_g = DDP(net_g, device_ids=[rank])
    # net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == which_device:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [
                               scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [
                               scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (x, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
        x = x.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(
            rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            # y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            y_hat, ids_slice, x_mask, z_mask,\
                (z, z_p, m_p, logs_p, m_q, logs_q), c1, c2 = net_g(
                    x, spec, spec_lengths, speakers)
            
            # slice to calc ctr
            c1, slice_idx = commons.rand_slice_segments(c1, spec_lengths, hps.train.segment_size // hps.data.hop_length)
            c1.squeeze_(1)
            c2 = commons.slice_segments(c2, slice_idx, hps.train.segment_size // hps.data.hop_length).squeeze(1)

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                  z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_ctr = constractive_loss(
                    c1.transpose(0, 1), c2.transpose(0, 1)) # constract for T dim
                # loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_ctr
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == which_device:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                # losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all,
                               "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                # scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl, "loss/g/ctr": loss_ctr})

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                    # "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(
                    hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(
                    hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1

    if rank == which_device:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
            x = x.cuda(which_device)
            spec, spec_lengths = spec.cuda(
                which_device), spec_lengths.cuda(which_device)
            y, y_lengths = y.cuda(which_device), y_lengths.cuda(which_device)
            speakers = speakers.cuda(which_device)

            # remove else
            x = x[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break
        # y_hat, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
        # x len same as y
        y_hat, mask, *_ = generator.infer(spec, spec_lengths, speakers)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
