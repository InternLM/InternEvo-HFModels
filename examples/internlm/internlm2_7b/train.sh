export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
export vmmDefragment=1

cd ./huggingface_model/internlm/internlm2_7b/
sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' modeling_internlm2.py
cd -

torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port=22500 ./examples/internlm/internlm2_7b/train.py --config ./examples/internlm/internlm2_7b/config.py --launcher "torch"
