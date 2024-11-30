nvidia-smi

# You can override the default parameters by passing variables to the script
base_model=${BASE:-"Llama-3.2-1B-Instruct"}
total=${BATCH:-2}      # total batch size
bs=${SEQ:-1}            # batch size per device
lr=${LR:-8e-4}
warmup_steps=${WU:-5000}
save_steps=${SAVE:-0.1}
num_gpus=${NUM_GPUS:-2}
segments_per_substep=${SEG:-2}
training_substeps=${SUB:-4}
summary_length=${SUM:-128}
num_nodes=${NUM_NODES:-1}
node=${NODE:-"localhost"}
summary_accumulation=${ACC:-true}
randomize_substeps=${RAND:-true}
segment_gradient_checkpointing=${CHECK:-true}
num_train_epochs=1

################################

total_per_device=$((${total}/${num_gpus}/${num_nodes}))
accu=$(( ${total_per_device} / ${bs} ))

run_name_suffix="sub${training_substeps}_seg${segments_per_substep}_sum${summary_length}_lr${lr}_bsz${total}_ft"
if [[ ${randomize_substeps} == true ]]; then
    run_name_suffix+="_rand"
fi
if [[ $summary_accumulation == true ]]; then
    run_name_suffix+="_accu"
fi
if [[ ${segment_gradient_checkpointing} == true ]]; then
    run_name+="_check"
fi
run_name="ac_${base_model}_${run_name_suffix}_ft"

echo "Run: ${run_name}"

cache_dir=./.cache
out_dir=checkpoints/$run_name
mkdir -p $out_dir

export OMP_NUM_THREADS=8
header="torchrun \
--nnodes=$num_nodes \
--nproc_per_node=$num_gpus \
--rdzv-backend=c10d \
--rdzv-endpoint=$node:5546 \
train.py "

model_url="checkpoints/ac_Llama-3.2-1B-Instruct_sub4_seg2_sum128_lr8e-4_bsz8_rand_accu"

arguments=(
    --report_to tensorboard
    --config_name $model_url
    --tokenizer_name $model_url
    --model_name_or_path $model_url
    --gradient_accumulation_steps $accu
    --per_device_eval_batch_size $bs
    --per_device_train_batch_size $bs
    --learning_rate $lr
    --warmup_steps $warmup_steps
    --do_train
    --logging_steps 1
    --save_steps $save_steps
    --save_total_limit 1
    --preprocessing_num_workers 6
    --dataloader_num_workers 6
    --cache_dir $cache_dir
    --add_special_tokens false
    --num_train_epochs ${num_train_epochs}
    --disable_tqdm true
    --resume_from_checkpoint true
    --log_level info
    --learning_rate $lr
    --run_name $run_name
    --output_dir $out_dir
    --summary_length $summary_length
    --accumulate_summary $summary_accumulation
    --remove_unused_columns false
    --segments_per_substep $segments_per_substep
    --training_substeps $training_substeps
    --randomize_substeps $randomize_substeps
    --segment_gradient_checkpointing $segment_gradient_checkpointing
    --bf16
    --use_fast_tokenizer false
    --chat_template llama-3
    $@
)

echo "Instruction Finetuning"
arguments+=(
    --min_length 2048
    --max_length 20480
    --train_data ../data/long-llm/gpt/one_detail_book.train.16K.json \
                 ../data/long-llm/gpt/one_detail_paper.train.16K.json \
                 ../data/long-llm/longalpaca/train.json \
                 ../data/long-llm/booksum/train.16K.json \
                 ../data/long-llm/needle/train.16K.json \
                 ../data/pretrain/llama3.2-8K_2B[5000]
)

#################

echo "Training ${base_model} with lr ${lr} on ${dataset}"
echo Outputting to $out_dir

echo command: echo "$header ${arguments[@]}"
$header ${arguments[@]} 2>&1 | tee -a $out_dir/log-resume.out
