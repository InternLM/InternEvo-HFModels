export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
export vmmDefragment=1

cd ./huggingface_model/Qwen/Qwen2_7B/
sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' modeling_qwen2.py
cd -

torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port=22500 ./examples/Qwen/Qwen2_7B/train.py --config ./examples/Qwen/Qwen2_7B/config.py --launcher "torch"
