

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
TOKEN_NUM=4096

    MAX_LEN=$((2* ${TOKEN_NUM}))
    # Default values
    MODEL_PATH="your path to the model"  # Add default model path
    DATATYPES=("aime2025" "math" "amc" "minerva" "olympiad_bench")




    OUTPUT_DIR="outputs/${MODEL_PATH}"  # Add default output directory

    

    # Echo the values for verification
    echo "Model Path: ${MODEL_PATH}"
    echo "Datasets: ${DATATYPES[@]}"
    echo "Output Directory: ${OUTPUT_DIR}"

    # Loop through all datatypes
    for DATA_TYPE in "${DATATYPES[@]}"; do
        python3 -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=2 \
            data.path=data/${DATA_TYPE}.parquet \
            data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
            data.n_samples=16 \
            data.batch_size=512 \
            model.path=${MODEL_PATH} \
            rollout.temperature=0.6 \
            rollout.response_length=$MAX_LEN \
            rollout.top_k=-1 \
            rollout.top_p=0.95 \
            rollout.gpu_memory_utilization=0.9 \
            rollout.tensor_model_parallel_size=2 \
            rollout.ignore_think_token=True \
            rollout.log_prob_micro_batch_size=2
    done




