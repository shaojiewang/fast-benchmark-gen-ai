#pytorch sdpa
CUDA_VISIBLE_DEVICES=1
pip install diffusers einops
python ./python/pth_attention.py  --batch_size 2 \
    --seqlen_q 4680 \
    --seqlen_kv 4680 \
    --num_head 40 \
    --head_dim 72 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --note "ta"

python ./python/pth_attention.py  --batch_size 40 \
    --seqlen_q 936 \
    --seqlen_kv 936 \
    --num_head 40 \
    --head_dim 72 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --note "sa"


python ./python/pth_attention.py  --batch_size 40 \
    --seqlen_q 936 \
    --seqlen_kv 256 \
    --num_head 40 \
    --head_dim 72 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --note "ca"

#fa

python ./python/pth_attention.py  --batch_size 2 \
    --seqlen_q 4680 \
    --seqlen_kv 4680 \
    --num_head 40 \
    --head_dim 72 \
    --use_flash_attn \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --note "ta"

python ./python/pth_attention.py  --batch_size 40 \
    --seqlen_q 936 \
    --seqlen_kv 936 \
    --num_head 40 \
    --head_dim 72 \
    --use_flash_attn \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --note "sa"


python ./python/pth_attention.py  --batch_size 40 \
    --seqlen_q 936 \
    --seqlen_kv 256 \
    --num_head 40 \
    --head_dim 72 \
    --use_flash_attn \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --note "ca"