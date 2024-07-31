"""
Train a diffusion model on images.
"""

import argparse

from masked_diffusion import dist_util, logger
from masked_diffusion.image_datasets import load_data
from masked_diffusion.resample import create_named_schedule_sampler
from masked_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
from masked_diffusion.train_utils_fid_2 import TrainLoop
from masked_diffusion import create_diffusion, model_and_diffusion_defaults, diffusion_defaults
import masked_diffusion.models as models_mdt

def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist_multinode(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    configs = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(configs)
    print(args)
    image_size = configs['image_size']
    latent_size = image_size // 8
    model = models_mdt.__dict__[args.model](input_size=latent_size, mask_ratio=args.mask_ratio, decode_layer=args.decode_layer)
    print(model)
    diffusion = create_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        batch_size_sample=args.batch_size_sample,
        num_samples=args.num_samples,
        num_sampling_steps=args.num_sampling_steps,
        clip_denoised=args.clip_denoised,
        use_ddim=args.use_ddim,
        class_cond=args.class_cond,
        cfg_scale=args.cfg_scale,
        cfg_cond=args.cfg_cond,
        lr_anneal_steps=args.lr_anneal_steps
    ).run_loop()


def create_argparser():
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
        mask_ratio=None,
        decode_layer=None,
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


if __name__ == "__main__":
    main()
