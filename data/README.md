# AutoCoA Training Data

This folder contains the datasets used for training models in the AutoCoA (Automatic generation of Chain-of-Action) framework. These datasets are designed to teach reasoning models when and how to use external tools during multi-step reasoning processes.

## Datasets Overview

The datasets in this folder were created using the HotpotQA training set as the base data source. Each dataset serves a specific purpose in the AutoCoA training pipeline:

| Filename | Number of Samples | Purpose |
|----------|-------------------|---------|
| `CoT_data_1k.jsonl` | 1,000 | Pure chain-of-thought reasoning without external tools |
| `CoA_data_5k.jsonl` | 5,000 | Chain-of-action reasoning with external tool usage |
| `CoT+A_data_1.5k.json` | 1,500 | Contrastive pairs for learning when to trigger actions |

## Data Formats

### CoT Data Format (`CoT_data_1k.jsonl`)

Each line contains a JSON object with the following structure:

```json
{
    "messages": [
        {"role": "user", "content": "Prompt"}, 
        {"role": "assistant", "content": "CoT"}
    ]
}
```

### CoA Data Format (`CoA_data_5k.jsonl`)

Each line contains a JSON object with the following structure:

```json
{
    "messages": [
        {"role": "user", "content": "Prompt"}, 
        {"role": "assistant", "content": "CoA"}
    ]
}
```

### CoT+A Data Format (`CoT+A_data_1.5k.json`)

This file contains JSON objects with contrastive pairs that demonstrate when to use actions:

```json

[
    {
        "conversation": [
            {
                "from": "human",
                "value": "Prompt + prefix"
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": ""
        },
        "rejected": {
            "from": "gpt",
            "value": ""
        }
    }
]
```

## Usage in Training Pipeline

These datasets are used in different stages of the AutoCoA training pipeline:

1. **SFT Stage 1 (CoT+A)**: Uses the 1,500 contrastive pairs from `CoT+A_data_1.5k.json` to teach the model when to trigger actions using contrastive learning at the step level.

2. **SFT Stage 2 & 3 (CoT+CoA)**: Uses a combination of 5,000 CoA samples from `CoA_data_5k.jsonl` and 1,000 CoT samples from `CoT_data_1k.jsonl` to teach the model how to execute actions and maintain pure reasoning capabilities.

The inclusion of pure CoT data alongside CoA data ensures that the model maintains its ability to handle questions that don't require external tool usage, thus achieving a balance between reasoning and action.
