
model_path=./checkpoints/our_continue_sft_videollava-7b-all_data

CUDA_VISIBLE_DEVICES=0 python -m videollava.serve.cli_multimodal \
    --model-path $model_path \
    --conv-mode v1 \
    --load-4bit