port=$(shuf -i25000-30000 -n1)

data_path='data'

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1  deepspeed --include localhost:0,1,2,3,4,5,6,7  --master_port $port videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path checkpoints/Video-LLaVA-7B-split  \
    --version v1 \
    --data_path ./scripts/v1_5/data_list.json \
    --image_folder ${data_path}/image \
    --image_tower LanguageBind_Image \
    --audio_folder ${data_path}/audio \
    --audio_tower LanguageBind_Audio \
    --video_folder ${data_path}/video \
    --video_tower LanguageBind_Video_merge \
    --depth_folder ${data_path}/depth \
    --depth_tower LanguageBind_Depth \
    --thermal_folder ${data_path}/thermal \
    --thermal_tower LanguageBind_Thermal \
    --electromagnetic_wave_folder ${data_path}/electromagnetic_wave \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter checkpoints/Video-LLaVA-7B-split/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/continue_sft_videollava-7b-${data} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"


