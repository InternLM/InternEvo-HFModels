export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
export vmmDefragment=1

cd ./huggingface_model/Yi/Yi_6B/
sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' modeling_yi.py
# patch modeling_yi.py internevo.patch
cd -

torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port=22500 ./examples/Yi/Yi_6B/train.py --config ./examples/Yi/Yi_6B/config.py --launcher "torch"