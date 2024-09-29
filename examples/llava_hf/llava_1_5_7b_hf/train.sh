export PYTHONPATH=./huggingface_model:$PYTHONPATH
export GPU_NUMS=8
srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/llava_hf/llava_1_5_7b_hf/train.py --config ./examples/llava_hf/llava_1_5_7b_hf/config.py