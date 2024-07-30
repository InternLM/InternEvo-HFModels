export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
srun -p $PARTITION -N $(expr $GPU_NUMS / 8) -n $GPU_NUMS --ntasks-per-node=8 --gpus-per-task=1 python ./examples/meta_llama/Meta_Llama_3_1_8B/train.py --config ./examples/meta_llama/Meta_Llama_3_1_8B/config.py