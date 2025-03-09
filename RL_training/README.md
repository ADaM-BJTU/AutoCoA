# VERL Modifications

In this project, we've made two major modifications to the VERL framework:

1. **Added a vllmrollout worker with search capabilities**
2. **Implemented observation mask functionality for the PPO trainer**

## GRPO training (via verl)

`run_grpo_sequence_parallel_stage1.sh` is the training code that uses the simulated environment.

`run_grpo_sequence_parallel_stage2.sh` is the launch script for using the real environment.
Before using it, please modify the `baseurl` in `verl/workers/rollout/vllm_rollout/vllm_rollout_coa.py` to match the IP address and port of FlashRAG.

## Evaluation

We have preprocessed the evaluation datasets. For evaluation, first use `model_merger.py` to convert the parameter files to HuggingFace format, then use `generate_response.sh` to generate responses for the test set questions, and finally use `eval_response.py` to evaluate performance.

