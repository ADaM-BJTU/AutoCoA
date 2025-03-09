
## Supervised Fine-Tuning

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning of the model. First, ensure your working directory is set to the current folder, then install the environment dependencies by running:

```bash
pip install -e ".[metrics,deepspeed]"
```

### SFT-Stage1

In this stage, prepare data that follows the CoT+A format (preference pair data). You can then perform step-level preference learning by running:

```bash
bash scripts/llamafty_ppf
```

### SFT-Stage2 & Stage3

For stages 2 and 3, run the corresponding supervised fine-tuning training using:

```bash
bash scripts/llamafty_sft.sh
```

Control whether tokens from external environment feedback are included in the loss calculation (learning objectives) using the `--ignore_observation` parameter in the script.
