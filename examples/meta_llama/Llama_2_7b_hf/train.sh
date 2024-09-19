export PYTHONPATH=./huggingface_model:$PYTHONPATH
export PARTITION="llm_s"
export GPU_NUMS=8
export vmmDefragment=1

cd ./huggingface_model/meta_llama/Llama_2_7b_hf/
sed -i 's/is_flash_attn_available/is_flash_attn_2_available/g' modeling_llama.py
cd -

torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port=22500 ./examples/meta_llama/Llama_2_7b_hf/train.py --config ./examples/meta_llama/Llama_2_7b_hf/config.py --launcher "torch"
