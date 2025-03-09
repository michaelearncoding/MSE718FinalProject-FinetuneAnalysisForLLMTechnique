# LLM Fine-tuning Analysis: A Comparative Study

## Overview

This repository contains the research and implementation for "A Comparative Analysis of Fine-tuning Techniques for Language Models," a project exploring different parameter-efficient fine-tuning methods across language models of varying sizes. Due to computational constraints, the current implementation focuses on Llama 3.2 (1B) and TinyLlama (1.1B) models.

## Repository Structure

```
├── evaluation_results/          # Evaluation outputs and metrics
├── lm-evaluation-harness/       # Benchmark evaluation framework
├── models/                      # Model weights and checkpoints
├── scripts/                     # Evaluation and analysis scripts
│   ├── analyze_results.py       # Processes and visualizes results
│   ├── evaluate_models.sh       # Main evaluation script
│   └── run_comparison.sh        # Comparison workflow
├── results/                     # Processed results and visualizations
│   ├── figures/                 # Generated charts and graphs
│   └── model_comparison/        # Comparison data
└── models_evaluation_record.md  # Detailed evaluation documentation
```

## Models Evaluated

| Model | Parameters | Training Data Size | Fine-tuning Method |
|-------|------------|-------------------|-------------------|


## Evaluation Benchmarks

The models were evaluated on the following tasks:

1. **GSM8K** - Mathematical reasoning (multi-step grade school math problems)
2. **HellaSwag** - Commonsense reasoning and situational understanding
3. **MMLU High School Computer Science** - Domain knowledge in computer science

## Setup and Usage

### Requirements

```
pip install -r requirements.txt
```

### Evaluation

To evaluate a model:

```bash
# Clone evaluation framework
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd ..

# Run evaluation
python -m lm_eval \
    --model hf \
    --model_args pretrained=PATH_TO_MODEL \
    --tasks hellaswag,gsm8k,mmlu_high_school_computer_science \
    --device cuda \
    --batch_size 4 \
    --output_path results/model_name.json
```

### Running Comparisons

```bash
# Full comparison pipeline
bash scripts/run_comparison.sh

# Individual model evaluation
bash scripts/evaluate_models.sh [model_size] [method] [merged]
```

## Research Focus

This project investigates several key research questions:

1. How do different parameter-efficient fine-tuning techniques affect model performance across various tasks?
2. What are the trade-offs between mathematical reasoning, commonsense reasoning, and domain knowledge during fine-tuning?
3. How does the base model architecture influence fine-tuning outcomes?
4. What impact does training data size have on fine-tuning efficiency?

Detailed results and analysis are available in the `models_evaluation_record.md` file.

## Conclusions

This research provides valuable insights into the trade-offs involved in fine-tuning smaller language models:

1. **Base Model Quality Matters**: The quality of the base architecture significantly impacts performance, with different architectures showing varying capabilities despite similar parameter counts.

2. **Fine-tuning Trade-offs**: Traditional fine-tuning can improve specific capabilities but often at the cost of others, showing interesting patterns of knowledge preservation and forgetting.

3. **Parameter-Efficient Methods**: LoRA and QLoRA show promise for preserving more base model capabilities, with QLoRA offering comparable performance with reduced memory requirements.

## Future Work

- [ ] Evaluate more parameter-efficient fine-tuning methods (LoRA, QLoRA) on Llama 3.2 1B
- [ ] Test mixed fine-tuning approaches to mitigate catastrophic forgetting
- [ ] Explore the impact of different fine-tuning dataset sizes on model performance
- [ ] Evaluate task-specific fine-tuning to optimize for particular capabilities
- [ ] Expand to larger models as computing resources become available

## Citation

If you use this code or findings in your research, please cite:

```
@mse{mai2025comparative,
  author = {Mai, Qingda},
  title = {A Comparative Analysis of Fine-tuning Techniques for Language Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/michaelearncoding/MSE718FinalProject-FinetuneAnalysisForLLMTechnique}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 