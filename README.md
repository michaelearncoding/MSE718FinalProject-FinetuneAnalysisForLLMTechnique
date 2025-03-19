# LLM Fine-tuning Analysis: A Comparative Study

This repository contains scripts and notebooks for evaluating and comparing different fine-tuning approaches on language models, specifically focusing on Llama 3.2 1B and TinyLlama 1.1B.

## Project Overview

This project investigates the effectiveness of different fine-tuning methods on language models, with a particular focus on:
- Parameter-efficient fine-tuning (PEFT) methods
- Impact of training data size on model performance
- Catastrophic forgetting in fine-tuned models
- Performance comparison across different model architectures

## Requirements

### Hardware Requirements
- GPU with at least 16GB VRAM (Tesla T4 or equivalent recommended)
- CUDA 12.4 or later
- NVIDIA drivers 550.54.15 or later

### Software Requirements
- Python 3.8+
- CUDA toolkit
- PyTorch
- Transformers library
- Unsloth library
- lm-evaluation-harness

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-peft-compare.git
cd llm-peft-compare
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install CUDA toolkit if not already installed:
```bash
# For Ubuntu/Debian
sudo apt-get install cuda-toolkit-12-4

# For macOS (using Homebrew)
brew install cuda
```

## Usage

### 1. Environment Setup

First, check your CUDA installation and GPU status:
```python
# Check CUDA version
!nvcc --version

# Check GPU status
!nvidia-smi
```

### 2. Model Evaluation

To evaluate a model using the lm-evaluation-harness:

```python
# For base model evaluation
python -m lm_eval --model hf \
    --model_args pretrained={MODEL_DIR},dtype=float16 \
    --tasks {TASKS} \
    --device cuda \
    --batch_size 4

# For fine-tuned model evaluation
python -m lm_eval --model hf \
    --model_args pretrained={MODEL_DIR} \
    --tasks {TASKS} \
    --device cuda \
    --batch_size 4
```

Replace `{MODEL_DIR}` with your model path and `{TASKS}` with your target tasks.

### 3. Available Tasks

The following tasks are supported:
- GSM8K (mathematical reasoning)
- HellaSwag (commonsense reasoning)
- MMLU High School Computer Science (domain knowledge)

### 4. Model Paths

Standard model paths used in the project:
- Base Llama 3.2 1B: `/content/llama-3.2-1b-base`
- Fine-tuned Llama 3.2 1B: `/content/llama-3.2-1b-merged`
- TinyLlama 1.1B: `/models/tinyllama_1.1b-instruction-lora-merged`

## Project Structure

```
llm-peft-compare/
├── notebooks/
│   ├── 0.Colab&UnSloth_llma3.2_1B_qlora_mergedModel(2k).ipynb
│   └── 1.Colab&Unsloth_llma3.2_evaluation.ipynb
├── scripts/
│   └── run_evaluation.sh
├── models_evaluation_record.md
└── requirements.txt
```

## Evaluation Results

Detailed evaluation results and comparisons can be found in `models_evaluation_record.md`.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm-peft-compare,
  author = {Your Name},
  title = {LLM Fine-tuning Analysis: A Comparative Study},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/llm-peft-compare}
}
```

## Acknowledgments

- Llama 3.2 model by Meta AI
- TinyLlama by the TinyLlama team
- Unsloth library for efficient fine-tuning
- lm-evaluation-harness for model evaluation 