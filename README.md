<div align="center">
<h1>Agent models: Internalizing Chain-of-Action Generation into Reasoning models</h1>
<a href="https://github.com/ADaM-BJTU/AutoCoA/blob/main/CoA_paper.pdf" target="_blank">
    <img src="https://img.shields.io/badge/PDF-Download-red?logo=adobeacrobatreader" alt="PDF">
</a>
<a href="https://arxiv.org/abs/your-arxiv-id" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv">
</a>
</div>

AutoCoA (Automatic generation of Chain-of-Action) is an agent model framework that enhances the multi-turn tool usage capability of reasoning models. The framework internalizes the Chain-of-Action (CoA) generation, allowing agent models to autonomously decide when and how to use external tools to improve task completion in open-domain QA tasks, particularly those requiring long-term reasoning and multi-step actions.


## Features
- **End-to-end Tuning:** Combines Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to enable seamless transition between reasoning and action.
- **Step-level Action Triggering:** Teaches the model when to take action using contrastive learning at critical reasoning junctures.
- **Trajectory-level CoA Optimization:** Enhances the model's behavior through full CoT (Chain-of-Thought) and CoA sequences.
- **Internal World Model:** Reduces interaction costs with real environments by simulating tool interactions during training.

## 📰 News
- 2025.03.09 — Code Released
- 2025.03.09 — Paper Released


## TODO
- Exploring more effective methods for internal world modeling or simulated environment usage.
- RFT in Tool Usage


## Acknowledge

This implementation draws heavily on [verl](https://github.com/volcengine/verl)  and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG). We extend our sincere gratitude to these projects for their valuable contributions to the open-source community.

<!--
****
## Citation

If this work is helpful to your research, please cite our paper:

<!-- ```
@article{
to 
}
``` -->
