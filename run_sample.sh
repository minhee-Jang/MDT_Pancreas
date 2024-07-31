
pip install -e .
# pip uninstall -y timm
# pip install mpi4py timm diffusers
#MODEL_PATH=output_mdt_xl2/mdt_xl2_v1_ckpt.pt
MODEL_PATH='/home/aix24705/data/MDT/output_mdt_s2/ema_0.9999_310000.pt'
export OPENAI_LOGDIR=output_mdt_xl2_eval
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
TF_ENABLE_ONEDNN_OPTS=0

'''
echo 'CFG Class-conditional sampling:'
MODEL_FLAGS="--image_size 512 --model MDT_S_2 --decode_layer 2"
DIFFUSION_FLAGS="--num_sampling_steps 250 --num_samples 10  --cfg_cond True"
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
 python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
# echo $MODEL_FLAGS
# echo $DIFFUSION_FLAGS
# echo $MODEL_PATH
# python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz
'''

echo Class-conditional sampling:
MODEL_FLAGS="--image_size 512 --model MDT_S_2 --decode_layer 2"
DIFFUSION_FLAGS="--num_sampling_steps 500 --num_samples 6000"     #6000 images
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH

#python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_5000x256x256x3.npz

