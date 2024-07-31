import copy
import functools
import os

import blobfile as bf
import torch as th

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from diffusers.models import AutoencoderKL

import argparse
import os
import tensorflow.compat.v1 as tf
import numpy as np
import torch as th
import torch.distributed as dist
from masked_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)
from evaluations.evaluator import FIDStatistics,Evaluator,ManifoldEstimator,DistanceBlock
from masked_diffusion import (
        create_diffusion,
        model_and_diffusion_defaults,
        diffusion_defaults, 
        dist_util,
        logger,
)
from evaluations.pancreas import *

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from masked_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)

from masked_diffusion import (
        create_diffusion,
        model_and_diffusion_defaults,
        diffusion_defaults, 
        dist_util,
        logger,
)

import masked_diffusion.models as models_mdt
from diffusers.models import AutoencoderKL
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model="MDT_S_2",
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        scale_factor=0.18215,
        batch_size_sample=16,
        num_samples=10,
        num_sampling_steps=500,
        clip_denoised=False,
        use_ddim=False,
        class_cond=True,
        cfg_scale=3.8,
        cfg_cond=False
    ):
        self.batch_size_sample=batch_size_sample
        self.num_samples=num_samples
        self.num_sampling_steps=num_sampling_steps
        self.clip_denoised=clip_denoised
        self.use_ddim=use_ddim
        self.class_cond=class_cond
        self.cfg_scale=cfg_scale
        self.cfg_cond=cfg_cond
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.scale_factor = scale_factor
        self.resume_step=140000
        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
         
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
        
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            #self.ddp_model = DDP(
            #    self.model,
            #    device_ids=[dist_util.dev()],
            #    output_device=dist_util.dev(),
            #    broadcast_buffers=False,
            #    bucket_cap_mb=128,
            #    find_unused_parameters=False,
            #)
        else:   
            if dist.get_world_size() > 1:
                    logger.warn(
                        "Distributed training requires CUDA. "
                        "Gradients will not be synchronized properly!"
                    )
            self.use_ddp = False
            self.ddp_model = self.model
        self.instantiate_first_stage()


    def instantiate_first_stage(self):
        model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dist_util.dev())
        self.first_stage_model = model.eval()
        self.first_stage_model.train = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    # https://github.com/huggingface/diffusers/blob/29b2c93c9005c87f8f04b1f0835babbcea736204/src/diffusers/models/autoencoder_kl.py
    @th.no_grad()
    def get_first_stage_encoding(self, x):
            encoder_posterior = self.first_stage_model.encode(x, return_dict=True)[0]

            z = encoder_posterior.sample()
            return z.to(dist_util.dev()) * self.scale_factor

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate): 
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    def create_argparser(self):
        defaults = dict(
            data_dir='/home/aix24705/data/MDT/datasets/2D_pancreas_cancer_crop_512/Train',
            schedule_sampler="uniform",
            lr=3e-4,
            weight_decay=0.0,
            lr_anneal_steps=0,
            batch_size=1,
            microbatch=-1,  # -1 disables microbatches
            ema_rate="0.9999",  # comma-separated list of EMA values
            log_interval=10,
            save_interval=10,
            resume_checkpoint="/home/aix24705/data/MDT/output_mdt_s2/ema_0.9999_140000.pt",
            use_fp16=False, 
            fp16_scale_growth=1e-3,
            model="MDT_S_2",
            mask_ratio=0.3,
            decode_layer=2,
            num_sampling_steps=250,
            clip_denoised=False,
            num_samples=10,
            batch_size_sample=16,
            use_ddim=False,
            model_path="",
            class_cond=True,
            cfg_scale=3.8,
            cfg_cond=False,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument(
            "--rank", default=0, type=int, help="""rank for distrbuted training."""
        )
        add_dict_to_argparser(parser, defaults)
        return parser
    def run_loop(self):
        self.resume_step=140000
        
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                print('fid_score....')
                th.backends.cuda.matmul.allow_tf32 = True
                args = create_argparser().parse_args()

                dist_util.setup_dist()
                logger.configure()

                logger.log("creating model and diffusion...")

                configs = args_to_dict(args, model_and_diffusion_defaults().keys())
                print(configs)
                image_size = configs['image_size']
                latent_size = image_size // 8
                model = models_mdt.__dict__[args.model](input_size=latent_size, decode_layer=args.decode_layer)
                msg = model.load_state_dict(
                    dist_util.load_state_dict(args.model_path, map_location="cpu")
                )
                print(msg)
                config_diffusion = args_to_dict(args, diffusion_defaults().keys())
                config_diffusion['timestep_respacing']= str(args.num_sampling_steps)
                print(config_diffusion)
                diffusion = create_diffusion(**config_diffusion)
                model.to(dist_util.dev())
                if args.use_fp16:
                    model.convert_to_fp16()
                model.eval()
                th.set_grad_enabled(False)

                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-"+str(args.vae_decoder)).to(dist_util.dev())

                logger.log("sampling...")
                all_images = []
                all_labels = []
                while len(all_images) * args.batch_size < args.num_samples:
                    model_kwargs = {}
                    if args.cfg_cond:
                        classes = th.randint(
                            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                        )
                        z = th.randn(args.batch_size, 4, latent_size, latent_size, device=dist_util.dev())
                        # Setup classifier-free guidance:
                        z = th.cat([z, z], 0)
                        classes_null = th.tensor([NUM_CLASSES] * args.batch_size, device=dist_util.dev())
                        classes_all = th.cat([classes, classes_null], 0)
                        model_kwargs["y"] = classes_all
                        model_kwargs["cfg_scale"] = args.cfg_scale
                        model_kwargs["diffusion_steps"] = config_diffusion['diffusion_steps']
                        model_kwargs["scale_pow"] = args.scale_pow
                    else:
                        z = th.randn(args.batch_size, 4, latent_size, latent_size, device=dist_util.dev())
                        classes = th.randint(
                            low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
                        )
                        
                        model_kwargs["y"] = classes


                    sample_fn = (
                        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                    )
                    sample = sample_fn(
                        model.forward_with_cfg,
                        z.shape,
                        z,
                        clip_denoised=args.clip_denoised,
                        progress=True, 
                        model_kwargs=model_kwargs,
                        device=dist_util.dev()
                    )
                    if args.cfg_cond:
                        sample, _ = sample.chunk(2, dim=0)  # Remove null class samples
                    # latent to image
                    sample = vae.decode(sample / 0.18215).sample
                    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) # clip in range -1,1
                    sample = sample.permute(0, 2, 3, 1)
                    sample = sample.contiguous()

                    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                    if args.class_cond:
                        gathered_labels = [
                            th.zeros_like(classes) for _ in range(dist.get_world_size())
                        ]
                        dist.all_gather(gathered_labels, classes)
                        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                    logger.log(f"created {len(all_images) * args.batch_size} samples")

                arr = np.concatenate(all_images, axis=0)
                arr = arr[: args.num_samples]
                if args.class_cond:
                    label_arr = np.concatenate(all_labels, axis=0)
                    label_arr = label_arr[: args.num_samples]
                
                #config = tf.ConfigProto(
                #allow_soft_placement=False  # allows DecodeJpeg to run on CPU in Inception graph
            #)
                
                #config.gpu_options.allow_growth = False
                #evaluator = Evaluator(tf.Session(config=config))
                evaluator = Evaluator(tf.Session())
                #evaluator.fid_sample=arr
                evaluator.fid_sample_array = arr
                evaluator.fid_label_array = label_arr
                evaluator.fid_sample_dataloader=torch.utils.data.DataLoader(evaluator.fid_sample_array,batch_size=32,shuffle=False)
                evaluator.fid_dataset=pancreas(data_dir='/home/aix24705/data/MDT/datasets/2D_pancreas_cancer_crop_512',state='train',im_size=(512,512),label=evaluator.fid_label_array)
                evaluator.fid_dataloader=torch.utils.data.DataLoader(evaluator.fid_dataset,batch_size=32,shuffle=False)
                ref_acts = evaluator.compute_activations(ref=True)
                ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_acts)
                sample_acts = evaluator.compute_activations(ref=False)
                sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_acts)
                print("Computing evaluations...")
                print("Inception Score:", evaluator.compute_inception_score(sample_acts[0]))
                print("FID:", sample_stats.frechet_distance(ref_stats))
                print("sFID:", sample_stats_spatial.frechet_distance(ref_stats_spatial))
                prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
                print("Precision:", prec)
                print("Recall:", recall)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):

            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro = self.get_first_stage_encoding(micro).detach()
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            micro_cond_mask = micro_cond.copy()
            micro_cond_mask['enable_mask']=True
            compute_losses_mask = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond_mask,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
                losses_mask = compute_losses_mask()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
                    losses_mask = compute_losses_mask()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach() + losses_mask["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean() + (losses_mask["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            log_loss_dict(
                self.diffusion, t, {'m_'+k: v * weights for k, v in losses_mask.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()
        


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
