export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=32
srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/mistralai/mixtral_8x7B_v0_1/train.py --config ./examples/mistralai/mixtral_8x7B_v0_1/config.py
