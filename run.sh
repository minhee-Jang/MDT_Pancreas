pip install -e .
export OPENAI_LOGDIR=output_mdt_s2
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
export 
MODEL_FLAGS="--image_size 512 --mask_ratio 0.30 --decode_layer 2 --model MDT_S_2"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size 16"
DATA_PATH='/home/aix24705/data/MDT/datasets/2D_pancreas_cancer_crop_512/Train'

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
#python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_train.py  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS