

cd src/open-r1-multimodal

var=0.11
noise=1.0
r=0.1
p_alpha=0.01

name=noisy_grpo_mmrlhf13k_img1536_r${r}_pa${p_alpha}_0369_3B
#name=debug
exps=output/${name}
log_dir=${exps}/output.log
mkdir -p ${exps}

export DEBUG_MODE=true
export LOG_PATH=${exps}/debug.log

torchrun --nproc_per_node=8 \
    --master_port=12346 \
    src/open_r1/grpo_hallucination_noise_self.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${exps} \
    --model_name_or_path [PATH_TO_Qwen/Qwen2.5-VL-3B-Instruct] \
    --dataset_name data_config/mmrlhf13k.yaml \
    --max_prompt_length 1024 \
    --num_generations 1 \
    --noise_extents 0.0 0.3 0.6 0.9 \
    --per_device_train_batch_size 2 \
    --p_alpha ${p_alpha} \
    --r_value ${r} \
    --noise_weight ${noise} \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --cot_reward_var ${var} \
    --bf16 \
    --torch_dtype bfloat16 \
    --reward_funcs correctness_bem_score format noisy \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --freeze_vision_modules true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --inner_gen_bs 8 \
    --run_name ${exps} \
    --save_steps 100 \
      --max_pixels 2359296 \
    --save_only_model true \
    2>&1 | tee -a ${log_dir}

