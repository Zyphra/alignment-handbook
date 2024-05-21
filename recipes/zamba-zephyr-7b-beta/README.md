
# Instructions to Replicate Zephyr-7b-β

As described in the Zephyr [technical report](https://huggingface.co/papers/2310.16944), training this model proceeds in two steps:

1. Apply SFT to fine-tune Zamba 7B on a filtered version of the UltraChat dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)). The result is an SFT model like [`zamba-zephyr-7b-sft-full`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full) or [`zamba-zephyr-7b-sft-qlora`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-qlora).
2. Align the SFT model to AI feedback via DPO on a preprocessed version of the UltraFeedback dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)). The result is an DPO model like `zamba-zephyr-7b-dpo-full` or `zamba-zephyr-7b-dpo-qlora`.

See below for commands to train these models using either DeepSpeed ZeRO-3 or LoRA.

## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model.
```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zamba-zephyr-7b-beta/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zamba-zephyr-7b-beta/dpo/config_full.yaml
```

## QLoRA training examples

Train faster with flash-attention 2 (GPU supporting FA2: A100, H100, etc)
```````shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zamba-zephyr-7b-beta/sft/config_qlora.yaml --load_in_4bit=true

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zamba-zephyr-7b-beta/dpo/config_qlora.yaml
```````

P.S. Using Flash Attention also allows you to drastically increase the batch size (x2 in my case)

Train without flash-attention:
```````shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zamba-zephyr-7b-beta/sft/config_qlora.yaml --load_in_4bit=true --use_flash_attention_2=false

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zamba-zephyr-7b-beta/dpo/config_qlora.yaml --use_flash_attention_2=false
```````