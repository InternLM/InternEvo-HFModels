export PYTHONPATH=./huggingface_model:$PYTHONPATH
export GPU_NUMS=8
srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/internlm/internlm2_7b/train.py --config ./examples/internlm/internlm2_7b/config.py