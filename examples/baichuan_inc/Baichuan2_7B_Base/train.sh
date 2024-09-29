export PYTHONPATH=./huggingface_model:$PYTHONPATH
export GPU_NUMS=8
srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/baichuan_inc/Baichuan2_7B_Base/train.py --config ./examples/baichuan_inc/Baichuan2_7B_Base/config.py