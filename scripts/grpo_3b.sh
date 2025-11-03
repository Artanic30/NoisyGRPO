export PATH=/2022233227/envs/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/2022233227/envs/cuda-12.4/lib64:$LD_LIBRARY_PATH

cd /2022233227/projects/RL_Hall

cd src/open-r1-multimodal

var=0.11
name=cot_vqa_mmrlhf13k_img1536_100_3b
#name=debug
exps=output/${name}
log_dir=${exps}/output.log
mkdir -p ${exps}

export DEBUG_MODE=true
export LOG_PATH=${exps}/debug.log

torchrun --nproc_per_node=8 \
    --master_port=12346 \
    src/open_r1/grpo_cot_vqa.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${exps} \
    --model_name_or_path /2022233227/pretrained_models/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/mmrlhf13k.yaml \
    --image_root /public/home/qiult/projects/BPO-main \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --reward_funcs correctness_bem_score format \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --freeze_vision_modules true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name ${exps} \
    --save_steps 100 \
      --max_pixels 2359296 \
    --save_only_model false \
    2>&1 | tee -a ${log_dir}
