#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p logs results

GPU_ENCODER="${GPU_ENCODER:-gpu:1}"
GPU_QWEN="${GPU_QWEN:-gpu:a100:1}"

MEM_ENCODER_BASE="${MEM_ENCODER_BASE:-32G}"
MEM_ENCODER_LARGE="${MEM_ENCODER_LARGE:-48G}"
MEM_QWEN_08B="${MEM_QWEN_08B:-64G}"
MEM_QWEN_4B="${MEM_QWEN_4B:-96G}"
MEM_QWEN_27B="${MEM_QWEN_27B:-160G}"

TIME_ENCODER_BASE_TRAIN="${TIME_ENCODER_BASE_TRAIN:-04:00:00}"
TIME_ENCODER_LARGE_TRAIN="${TIME_ENCODER_LARGE_TRAIN:-08:00:00}"
TIME_ENCODER_INFER="${TIME_ENCODER_INFER:-03:00:00}"
TIME_QWEN_08B_ZS="${TIME_QWEN_08B_ZS:-1-00:00:00}"
TIME_QWEN_4B_ZS="${TIME_QWEN_4B_ZS:-3-00:00:00}"
TIME_QWEN_27B_ZS="${TIME_QWEN_27B_ZS:-3-00:00:00}"
TIME_QWEN_08B_TRAIN="${TIME_QWEN_08B_TRAIN:-12:00:00}"
TIME_QWEN_4B_TRAIN="${TIME_QWEN_4B_TRAIN:-24:00:00}"
TIME_QWEN_27B_TRAIN="${TIME_QWEN_27B_TRAIN:-48:00:00}"
TIME_QWEN_08B_INFER="${TIME_QWEN_08B_INFER:-1-00:00:00}"
TIME_QWEN_4B_INFER="${TIME_QWEN_4B_INFER:-3-00:00:00}"
TIME_QWEN_27B_INFER="${TIME_QWEN_27B_INFER:-3-00:00:00}"

submit_job() {
    local description="$1"
    shift
    echo "Submitting ${description}..." >&2
    local job_id
    job_id="$(sbatch --parsable "$@")"
    job_id="${job_id%%;*}"
    echo "${description}: ${job_id}" >&2
    printf "%s" "${job_id}"
}

join_by_colon() {
    local IFS=:
    printf "%s" "$*"
}

all_inference_jobs=()

enc_base_train="$(submit_job "deberta-base train" \
    --job-name=ner-enc-base-train \
    --gres="${GPU_ENCODER}" \
    --mem="${MEM_ENCODER_BASE}" \
    --time="${TIME_ENCODER_BASE_TRAIN}" \
    scripts/cluster/job_encoder_train.sh configs/deberta_base.yaml)"
enc_base_infer="$(submit_job "deberta-base infer" \
    --job-name=ner-enc-base-infer \
    --dependency="afterok:${enc_base_train}" \
    --gres="${GPU_ENCODER}" \
    --mem="${MEM_ENCODER_BASE}" \
    --time="${TIME_ENCODER_INFER}" \
    scripts/cluster/job_encoder_infer.sh configs/deberta_base.yaml results/multinerd/deberta-v3-base/best_model)"
all_inference_jobs+=("${enc_base_infer}")

enc_large_train="$(submit_job "deberta-large train" \
    --job-name=ner-enc-large-train \
    --gres="${GPU_ENCODER}" \
    --mem="${MEM_ENCODER_LARGE}" \
    --time="${TIME_ENCODER_LARGE_TRAIN}" \
    scripts/cluster/job_encoder_train.sh configs/deberta_large.yaml)"
enc_large_infer="$(submit_job "deberta-large infer" \
    --job-name=ner-enc-large-infer \
    --dependency="afterok:${enc_large_train}" \
    --gres="${GPU_ENCODER}" \
    --mem="${MEM_ENCODER_LARGE}" \
    --time="${TIME_ENCODER_INFER}" \
    scripts/cluster/job_encoder_infer.sh configs/deberta_large.yaml results/multinerd/deberta-v3-large/best_model)"
all_inference_jobs+=("${enc_large_infer}")

zs_08b="$(submit_job "qwen35-08b zeroshot" \
    --job-name=ner-zs-08b \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_08B}" \
    --time="${TIME_QWEN_08B_ZS}" \
    scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_08b_zeroshot.yaml)"
all_inference_jobs+=("${zs_08b}")

zs_4b="$(submit_job "qwen35-4b zeroshot" \
    --job-name=ner-zs-4b \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_4B}" \
    --time="${TIME_QWEN_4B_ZS}" \
    scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_4b_zeroshot.yaml)"
all_inference_jobs+=("${zs_4b}")

zs_27b="$(submit_job "qwen35-27b zeroshot" \
    --job-name=ner-zs-27b \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_27B}" \
    --time="${TIME_QWEN_27B_ZS}" \
    scripts/cluster/job_decoder_zeroshot.sh configs/qwen35_27b_zeroshot.yaml)"
all_inference_jobs+=("${zs_27b}")

lora_08b_train="$(submit_job "qwen35-08b lora train" \
    --job-name=ner-lora-08b-train \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_08B}" \
    --time="${TIME_QWEN_08B_TRAIN}" \
    scripts/cluster/job_decoder_lora_train.sh configs/qwen35_08b.yaml)"
lora_08b_infer="$(submit_job "qwen35-08b lora infer" \
    --job-name=ner-lora-08b-infer \
    --dependency="afterok:${lora_08b_train}" \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_08B}" \
    --time="${TIME_QWEN_08B_INFER}" \
    scripts/cluster/job_decoder_lora_infer.sh configs/qwen35_08b.yaml results/multinerd/qwen35-08b-qlora/best_lora_adapter)"
all_inference_jobs+=("${lora_08b_infer}")

lora_4b_train="$(submit_job "qwen35-4b lora train" \
    --job-name=ner-lora-4b-train \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_4B}" \
    --time="${TIME_QWEN_4B_TRAIN}" \
    scripts/cluster/job_decoder_lora_train.sh configs/qwen35_4b.yaml)"
lora_4b_infer="$(submit_job "qwen35-4b lora infer" \
    --job-name=ner-lora-4b-infer \
    --dependency="afterok:${lora_4b_train}" \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_4B}" \
    --time="${TIME_QWEN_4B_INFER}" \
    scripts/cluster/job_decoder_lora_infer.sh configs/qwen35_4b.yaml results/multinerd/qwen35-4b-qlora/best_lora_adapter)"
all_inference_jobs+=("${lora_4b_infer}")

lora_27b_train="$(submit_job "qwen35-27b lora train" \
    --job-name=ner-lora-27b-train \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_27B}" \
    --time="${TIME_QWEN_27B_TRAIN}" \
    scripts/cluster/job_decoder_lora_train.sh configs/qwen35_27b.yaml)"
lora_27b_infer="$(submit_job "qwen35-27b lora infer" \
    --job-name=ner-lora-27b-infer \
    --dependency="afterok:${lora_27b_train}" \
    --gres="${GPU_QWEN}" \
    --mem="${MEM_QWEN_27B}" \
    --time="${TIME_QWEN_27B_INFER}" \
    scripts/cluster/job_decoder_lora_infer.sh configs/qwen35_27b.yaml results/multinerd/qwen35-27b-qlora/best_lora_adapter)"
all_inference_jobs+=("${lora_27b_infer}")

compare_dep="$(join_by_colon "${all_inference_jobs[@]}")"
compare_job="$(submit_job "comparison" \
    --job-name=ner-compare \
    --dependency="afterok:${compare_dep}" \
    --mem=8G \
    --time=00:30:00 \
    scripts/cluster/job_compare.sh results)"

echo
echo "Submitted all jobs."
echo "Inference jobs: ${all_inference_jobs[*]}"
echo "Comparison job: ${compare_job}"
