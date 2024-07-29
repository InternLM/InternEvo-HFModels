export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/yi/yi_6B/train.py --port 8352 --config ./examples/yi/yi_6B/config.py