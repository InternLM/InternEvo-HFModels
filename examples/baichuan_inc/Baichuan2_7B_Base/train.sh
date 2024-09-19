export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
export vmmDefragment=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2920

cd ./huggingface_model/baichuan_inc/Baichuan2_7B_Base/
sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' modeling_baichuan.py
cd -

torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port=22500 ./examples/baichuan_inc/Baichuan2_7B_Base/train.py --config ./examples/baichuan_inc/Baichuan2_7B_Base/config.py --launcher "torch"
