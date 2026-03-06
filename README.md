# SDPO using Tinker API

[Reinforcement Learning via Self-Distillation paper](https://arxiv.org/pdf/2601.19897) (SDPO) proposes on-onlicy distilation method that relies on in-context learning abilities of the model by using the same model re-prompted with extra context (demonstration\feedback) as a teacher, to provide a dense reward for grading stundent's trajectories.

It is not clear how tuned the final system is and how much hyperparameter sweep & reporting best validation accuracy (E.2.1) is responsible for the gains of the method
 - teacher regularization (Trust-region, EMA and alpha in Table 4 & E.2) vs using fixed reference model
 - reverse KL divergence vs Jensen–Shannon

This research prototype loosly reproduces Fig.1 plot in simpler settings using using only
 - LoRA fine-tuning using Tinker API
 - LiveCodeBenchV6 (\w same split for training and holdout tests)
 - reverse KL divergence as loss function
 - token and logit-level distillation


## Intermediate results

* LiveCodeBenchV6 [tests are split](https://huggingface.co/datasets/bzz2/live_code_bench_v6_lite_sdpo) into training and holdout sets using [replication package](https://github.com/lasgroup/SDPO/compare/main...bzz:SDPO:claude/generate-lcb-v6-dataset-ZQdUl) from the paper.

* Visualize logprob difference for teacher and student responses for a few samples _LiveCodeBenchV5 from code_rl recipe_:
```sh
python play_w_code_env.py --n-tasks 3 --model Qwen/Qwen3-8B -n 4 --seed 44
```

## Steps

For execution feedback we use [Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) provides local Docker-based sandboxing. 


1. Start a local sandbox in Docker with:
```sh
docker run -it -p 8080:8080 \
    -v ${PWD}/sandbox_config_local.yaml:/root/sandbox/sandbox/configs/local.yaml \
    volcengine/sandbox-fusion:server-20250609
export SANDBOX_URL=http://localhost:8080/run_code
```

2. Train a `Qwen3-4B-Instruct-2507` model with:
```sh
python -m sdpo_on_policy_distillation \
    model_name='Qwen/Qwen3-4B-Instruct-2507' \
    dataset=lcb \
    group_size=8 groups_per_batch=128 \
    learning_rate=4e-5 \
    lora_rank=32 \
    max_tokens=24576 \
    wandb_project=sdpo_lcb_distillation
```


3. Evaluate the model on holdout tests from LiveCodeBenchV6 with:
TODO: find the simplest way to run evaluation of the trained model on holdout split of tests from LiveCodeBenchV6 using Sandbox Fusion for isolation
```sh
```

Resources
 - Original [SDPO paper](sdpo-paper.pdf)
 - [Tinker Cookbook](tinker-llms-full.txt)
 - Example of [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) recipie
