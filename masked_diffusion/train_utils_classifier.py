import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from diffusers.models import AutoencoderKL
import tensorflow.compat.v1 as tf
import argparse
import os
from evaluations.evaluator import FIDStatistics,Evaluator,ManifoldEstimator,DistanceBlock
import numpy as np
import torch as th
import torch.distributed as dist
from masked_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)
from evaluations.pancreas import pancreas
from masked_diffusion import (
        create_diffusion,
        model_and_diffusion_defaults,
        diffusion_defaults, 
        dist_util,
        logger
)
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
from evaluations.eval import *
import masked_diffusion.models as models_mdt
from diffusers.models import AutoencoderKL
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)

def calculate_fid(loader1,loader2):
    #print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    #self.loader1=loader1
    #self.loader2=loader2
    loaders = [loader1,loader2]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            x=x.float().to('cuda')
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        test_data,
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
        scale_factor=0.18215, # scale_factor follows DiT and stable diffusion.
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.test_data=test_data
      
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
            num_sampling_steps=500,
            clip_denoised=False,
            num_samples=5000,
            batch_size=16,
            use_ddim=False,
            model_path="/home/aix24705/data/MDT/output_mdt_s2/ema_0.9999_140000.pt",
            model="MDT_S_2",
            class_cond=True,
            cfg_scale=3.8,
            decode_layer=2,
            cfg_cond=False,
        )

        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        parser.add_argument('--scale_pow', default=4, type=float)
        parser.add_argument('--vae_decoder', type=str, default='ema')  # ema or mse
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        #parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument(
            "--rank", default=0, type=int, help="""rank for distrbuted training."""
        )
        parser.add_argument('--data_dir', default='/home/aix24705/data/MDT/datasets/2D_pancreas_cancer_crop_512/Train',
                            help='url used to set up distributed training')
        add_dict_to_argparser(parser, defaults)
        return parser


    def run_loop(self):
        self.resume_step=300000
        
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            for i, data in enumerate(self.data):
                batch, cond = data
                batch=batch.requires_grad_(True)
                #self.mp_trainer.zero_grad()
                th.set_grad_enabled(True)
                #cond=cond.requires_grad_(True)
                self.run_step(batch, cond)

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    #if self.step != 0:
                    self.save()
                    #if self.step != 0:
                    print('fid_score....')
                    #th.backends.cuda.matmul.allow_tf32 = True
                    args = self.create_argparser().parse_args()
                    dist_util.setup_dist()
                    logger.configure()
                    logger.log("creating model and diffusion...")

                    configs = args_to_dict(args, model_and_diffusion_defaults().keys())
                    print(configs)
                    image_size = configs['image_size']
                    latent_size = image_size // 8
                    
                    eval_model = models_mdt.__dict__[args.model](input_size=latent_size, decode_layer=args.decode_layer)
                    msg = eval_model.load_state_dict(
                        dist_util.load_state_dict('/home/aix24705/data/MDT/output_mdt_s2/ema_0.9999_{}.pt'.format(self.resume_step+self.step), map_location="cpu")
                    )
                    print(msg)
                    config_diffusion = args_to_dict(args, diffusion_defaults().keys())
                    config_diffusion['timestep_respacing']= str(args.num_sampling_steps)
                    print(config_diffusion)
                    diffusion = create_diffusion(**config_diffusion)
                    eval_model.to(dist_util.dev())
                    if args.use_fp16:
                        eval_model.convert_to_fp16()
                    eval_model.eval()
                    th.set_grad_enabled(False)
                    #test_batch, test_cond = next(self.test_data)
                    class_embedding=[[i for j in range(args.batch_size)] for i in range(3)]  
                    loss_list=[]
                    acc=0.0
                    w_n_b=0.0
                    #test_batch, test_cond = next(self.test_data)
                    loader = torch.utils.data.DataLoader(
                    self.test_data, batch_size=16, shuffle=True, num_workers=1, drop_last=True
                    )
                    for i, data in enumerate(loader):
                    #for i in range(810):
                        test_batch, test_cond = data
                        test_batch=test_batch.to(dist_util.dev())
                        n_b=test_batch.shape[0]
                        #test_cond=data[1]
                        micro = test_batch
                        micro = self.get_first_stage_encoding(micro).detach()
                        micro_cond_real=test_cond
                        real_label=micro_cond_real['y']
                        for j in range(3):
                            micro_cond={'y':torch.tensor(np.array(class_embedding[j])).to(dist_util.dev())}
                            last_batch = (i + self.microbatch) >= test_batch.shape[0]
                            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                            
                            compute_losses = functools.partial(
                                self.diffusion.training_losses,
                                eval_model,
                                micro,
                                t,
                                model_kwargs=micro_cond,
                            )
                            
                            
                            micro_cond_mask = micro_cond.copy()
                            micro_cond_mask['enable_mask']=True
                            compute_losses_mask = functools.partial(
                                self.diffusion.training_losses,
                                eval_model,
                                micro,
                                t,
                                model_kwargs=micro_cond_mask,
                            )

                            if last_batch or not self.use_ddp:
                                losses = compute_losses()
                                losses_mask = compute_losses_mask()
                            else:
                                with eval_model.no_sync():
                                    losses = compute_losses()
                                    losses_mask = compute_losses_mask()
                            
                            if isinstance(self.schedule_sampler, LossAwareSampler):
                                self.schedule_sampler.update_with_local_losses(
                                    t, losses["loss"].detach() + losses_mask["loss"].detach()
                                )

                            loss = (losses["loss"] * weights) + (losses_mask["loss"] * weights)
                            loss_list.append(loss)
                        index_list=[]
                        for m in range(16):
                            n_l=[]
                            for n in range(3):
                                n_l.append(loss_list[n][m])
                            index_list.append(n_l.index(min(n_l)))
                        #index_list=torch.tensor(np.array(index_list))
                        correct = (np.array(index_list) == real_label.detach().cpu().numpy()).sum().item()
                        print(correct)
                        acc += correct
                        w_n_b += n_b
                    
                    print(f'Top 1 acc: {(acc/w_n_b) * 100:.2f}%') 
                    exit()
                            
                    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-"+str(args.vae_decoder)).to(dist_util.dev())

                    logger.log("sampling...")
                    all_images = []
                    all_labels = []
                    while len(all_images) * args.batch_size < args.num_samples:
                        e_model_kwargs = {}
                        if args.cfg_cond:
                            classes = th.randint(
                                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                            )
                            z = th.randn(args.batch_size, 4, latent_size, latent_size, device=dist_util.dev())
                            # Setup classifier-free guidance:
                            z = th.cat([z, z], 0)
                            classes_null = th.tensor([NUM_CLASSES] * args.batch_size, device=dist_util.dev())
                            classes_all = th.cat([classes, classes_null], 0)
                            e_model_kwargs["y"] = classes_all
                            e_model_kwargs["cfg_scale"] = args.cfg_scale
                            e_model_kwargs["diffusion_steps"] = config_diffusion['diffusion_steps']
                            e_model_kwargs["scale_pow"] = args.scale_pow
                        else:
                            z = th.randn(args.batch_size, 4, latent_size, latent_size, device=dist_util.dev())
                            classes = th.randint(
                                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                            )
                            
                            e_model_kwargs["y"] = classes


                        sample_fn = (
                            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                        )
                        sample = sample_fn(
                            eval_model.forward_with_cfg,
                            z.shape,
                            z,
                            clip_denoised=args.clip_denoised,
                            progress=True, 
                            model_kwargs=e_model_kwargs,
                            device=dist_util.dev()
                        )
                        if args.cfg_cond:
                            sample, _ = sample.chunk(2, dim=0)  # Remove null class samples
                        # latent to image
                        sample = vae.decode(sample / 0.18215).sample
                        sample = ((sample + 1) * 127.5).clamp(0, 255)
        
                        
                        
                        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                        

                        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                        all_images.extend([(sample.cpu().numpy()) for sample in gathered_samples])
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
                    
                    fid_sample_array = arr
                    fid_label_array = label_arr
                    fid_sample_dataloader=torch.utils.data.DataLoader(fid_sample_array,batch_size=25,shuffle=False)
                    fid_dataset=pancreas(data_dir='/home/aix24705/data/MDT/datasets/2D_pancreas_cancer_crop_512',state='train',im_size=(512,512),label=fid_label_array)
                    fid_dataloader=torch.utils.data.DataLoader(fid_dataset,batch_size=25,shuffle=False)
                    fid_score=calculate_fid(fid_sample_dataloader,fid_dataloader)
                    print('FID:{}'.format(fid_score))
                    
                self.step += 1
            
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        #batch.requires_grad_(True)
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        #batch=batch.requires_grad_(True)
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
            loss=loss.requires_grad_(True)
            
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
