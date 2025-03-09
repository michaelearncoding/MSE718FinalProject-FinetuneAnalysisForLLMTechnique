# Model Evaluation Results Record

## 1. Llama 3.2 Models (1.2B)

### Base Llama 3.2 1B

| Task | Metric | Result | Error Range |
|------|------|------|----------|
| **MMLU High School Computer Science** | Accuracy | **47.00%** | ±5.02% |
| HellaSwag | Accuracy | 45.23% | ±0.50% |
| HellaSwag | Normalized Accuracy | 60.77% | ±0.49% |
| GSM8K | Flexible Extract | 33.51% | ±1.30% |
| GSM8K | Strict Match | 33.51% | ±1.30% |

- **Evaluation Date**: 2025-03-08
- **Model Path**: `/content/llama-3.2-1b-base`
- **Evaluation Command**: `python -m lm_eval --model hf --model_args pretrained={MODEL_DIR},dtype=float16 --tasks {TASKS} --device cuda --batch_size 4`
- **Device**: CUDA (Tesla T4)
- **Batch Size**: 4
- **Data Type**: float16

### Fine-tuned Llama 3.2 1B (2K Examples)

| Task | Metric | Result | Error Range |
|------|------|------|----------|
| **MMLU High School Computer Science** | Accuracy | **34.00%** | ±4.76% |
| HellaSwag | Accuracy | 45.61% | ±0.50% |
| HellaSwag | Normalized Accuracy | 61.20% | ±0.49% |
| GSM8K | Flexible Extract | 3.71% | ±0.52% |
| GSM8K | Strict Match | 3.11% | ±0.48% |

- **Evaluation Date**: 2025-03-08
- **Model Path**: `/content/llama-3.2-1b-merged`
- **Evaluation Command**: `python -m lm_eval --model hf --model_args pretrained={MODEL_DIR} --tasks {TASKS} --device cuda --batch_size 4`
- **Fine-tuning Method**: Unsloth
- **Training Data Size**: <2,000 examples
- **Device**: CUDA (Tesla T4)
- **Batch Size**: 4
- **Data Type**: float16

## 2. TinyLlama (1.1B)

### TinyLlama + LoRA (50K Examples)

| Task | Metric | Result | Error Range |
|------|------|------|----------|
| **MMLU High School Computer Science** | Accuracy | **27.00%** | ±4.46% |
| HellaSwag | Accuracy | 45.84% | ±0.50% |
| HellaSwag | Normalized Accuracy | 59.26% | ±0.49% |
| GSM8K | Flexible Extract | 2.81% | ±0.45% |
| GSM8K | Strict Match | 2.20% | ±0.40% |

- **Evaluation Date**: 2025-03-07
- **Model Path**: `/models/tinyllama_1.1b-instruction-lora-merged`
- **Evaluation Command**: `./scripts/run_evaluation.sh tiny lora true`
- **Fine-tuning Method**: LoRA
- **Training Data Size**: Approximately 50,000 examples (Alpaca dataset)
- **Device**: MPS (Apple Silicon)
- **Batch Size**: 8
- **Data Type**: float16

### TinyLlama + QLoRA

| Task | Metric | Result | Error Range |
|------|------|------|----------|
| **MMLU High School Computer Science** | Accuracy | **26.50%** | ±4.42% |
| HellaSwag | Accuracy | 44.98% | ±0.50% |
| HellaSwag | Normalized Accuracy | 58.35% | ±0.49% |
| GSM8K | Flexible Extract | 2.74% | ±0.44% |
| GSM8K | Strict Match | 2.15% | ±0.39% |

- **Evaluation Date**: 2025-03-07
- **Model Path**: `/models/tinyllama_1.1b-instruction-qlora-merged`
- **Evaluation Command**: `./scripts/run_evaluation.sh tiny qlora true`
- **Device**: MPS (Apple Silicon)
- **Batch Size**: 8
- **Data Type**: float16

### TinyLlama Base Model (Not Evaluated)

*Not yet evaluated, planned using command: `./scripts/run_evaluation.sh base tiny false`*

## 3. Phi-2 (2.7B)

*Not yet evaluated, planned using command: `./scripts/run_evaluation.sh small qlora true`*

## 4. Mistral (7B)

*Not yet evaluated, planned using command: `./scripts/run_evaluation.sh medium qlora true`*

## Comparative Analysis

### Model Size Comparison

| Model | Parameters | Training Data Size |
|-------|------------|-------------------|
| Llama 3.2 1B | 1.24 billion | N/A (base) or 2K (fine-tuned) |
| TinyLlama 1.1B | 1.10 billion | 50K |

### Performance Comparison Table

| Task | Metric | Base Llama 3.2 1B | Fine-tuned Llama 3.2 1B (2k) | TinyLlama 1.1B (50k) |
|------|--------|-------------------|------------------------------|----------------------|
| **GSM8K** | Flexible Extract | **33.51% ± 1.30%** | 3.71% ± 0.52% | 2.81% ± 0.45% |
| **GSM8K** | Strict Match | **33.51% ± 1.30%** | 3.11% ± 0.48% | 2.20% ± 0.40% |
| **HellaSwag** | Accuracy | 45.23% ± 0.50% | 45.61% ± 0.50% | **45.84% ± 0.50%** |
| **HellaSwag** | Normalized Accuracy | 60.77% ± 0.49% | **61.20% ± 0.49%** | 59.26% ± 0.49% |
| **MMLU CS** | Accuracy | **47.00% ± 5.02%** | 34.00% ± 4.76% | 27.00% ± 4.46% |

### Key Observations

1. **Base Model Strength**: The base Llama 3.2 1B model significantly outperforms both its fine-tuned version and TinyLlama on GSM8K (mathematical reasoning) and MMLU CS (domain knowledge).

2. **Catastrophic Forgetting**: The fine-tuned Llama 3.2 1B model shows evidence of catastrophic forgetting, with significant regression in mathematical reasoning (-29.8 percentage points on GSM8K) and domain knowledge (-13.0 percentage points on MMLU CS).

3. **Commonsense Reasoning Preservation**: All models perform similarly on HellaSwag, suggesting that commonsense reasoning capabilities are more resistant to changes during fine-tuning.

4. **Training Data Efficiency**: While TinyLlama was trained on 25x more examples (50K vs 2K), the fine-tuned Llama 3.2 1B still outperforms it on MMLU CS and achieves similar or better results on other metrics, demonstrating the superior foundation of the Llama 3.2 architecture.

---

## Task Descriptions

1. **MMLU High School Computer Science**: Measures model performance on high school computer science knowledge questions, testing fundamental computer science concepts through multiple-choice questions.

2. **HellaSwag**: Measures commonsense reasoning and situational understanding, requiring models to select the most plausible sentence endings.
   - `Accuracy`: Standard accuracy
   - `Normalized Accuracy`: Accuracy after normalizing for text format variations

3. **GSM8K**: Measures mathematical reasoning ability, containing multi-step elementary and middle school math word problems.
   - `Flexible Extract`: Allows extracting answers from generated text
   - `Strict Match`: Requires exact format matching for answers

---

## Future Work

- [ ] Evaluate TinyLlama base model
- [ ] Evaluate Phi-2 QLoRA model
- [ ] Evaluate Mistral QLoRA model
- [ ] Evaluate more parameter-efficient fine-tuning methods (LoRA, QLoRA) on Llama 3.2 1B
- [ ] Test mixed fine-tuning approaches to mitigate catastrophic forgetting
- [ ] Compare performance across different model sizes with the same fine-tuning approach
- [ ] Compare performance of the same model with different fine-tuning approaches
- [ ] Explore the impact of different fine-tuning dataset sizes on model performance
- [ ] Evaluate task-specific fine-tuning to optimize for particular capabilities
