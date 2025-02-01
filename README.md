# CodeT5 Java Code Fine-Tuning

A project for fine-tuning Salesforce's CodeT5 model on Java code completion tasks using the CodeSearchNet dataset.

![CodeT5 Architecture](https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/codet5p_overview.png)*Example code structure (image from original CodeT5 repo)*

CodeT5+ paper and model.
[paper](https://arxiv.org/pdf/2305.07922.pdf) | [code](https://github.com/salesforce/CodeT5/tree/main/CodeT5+) | [model](https://huggingface.co/models?sort=downloads&search=codet5p)

## Features

- **Code Completion Fine-Tuning**: Adapted CodeT5 for Java code completion tasks
- **Checkpoint Management**: Automatic saving and resumption capabilities
- **Custom Training Callbacks**:
  - Memory management
  - Training progress samples
  - Automatic logging
- **Flexible Configuration**: CLI arguments for all training parameters
- **Evaluation Metrics**: Exact match comparison for generated code

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/grodondo/codet5-java-finetuning.git
cd codet5-java-finetuning-autocomplete-ai
```

2. **Create Virtual Enviroment (Optional)**
```bash
python -m venv /path/to/new/virtual/environment
```
- To open the virtual enviroment execute the following command:
```bash
# On Windows (Command Prompt)
.\path\to\new\virtual\environment\Scripts\activate

# On Windows (PowerShell)
.\path\to\new\virtual\environment\Scripts\Activate.ps1

# On macOS & Linux (Bash/Zsh)
source /path/to/new/virtual/environment/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
```bash
python train.py \
  --model_name Salesforce/codet5-small \
  --dataset_language java \
  --output_dir ./models \
  --batch_size 8 \
  --epochs 10 \
  --no-fp16     # --> for cpu training
```

### Resume Training
```bash
# Start training from the beginning
python train.py 

# Specify specific checkpoint
python train.py --resume_from_checkpoint ./models/checkpoint-12000
```

### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Salesforce/codet5-small` | Base pre-trained model |
| `--batch_size` | 4 | Training/evaluation batch size |
| `--learning_rate` | 3e-5 | Initial learning rate |
| `--epochs` | 5 | Number of training epochs |
| `--no-fp16` | True | Deactivate the mixed precision training |
| `--output_dir` | ./codet5-java | Output directory |
| `--resume_from_checkpoint` | False | resume from checkpoint-nnnnn |

## Training Process

### Checkpoint System
- Automatic saving after each epoch
- Retains 3 latest checkpoints
- Checkpoint format: `checkpoint-<step_number>`
  - Contains full model state + optimizer + scheduler

### Monitoring
- Real-time metrics in terminal
- TensorBoard logs in `./logs`
- Generated samples in `training_samples.txt`
- Memory management through automatic garbage collection

## Customization

### Dataset
- Default: CodeSearchNet Java split
- To modify:
  1. Implement custom dataset loader
  2. Update `preprocess_dataset()` in `train.py`
  3. Maintain `prompt`/`completion` format

### Model Architecture
```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "Salesforce/codet5-small",
    use_safetensors=True
)
```

## Troubleshooting

### Common Issues
- **Missing Checkpoints**: Verify `output_dir` permissions
- **CUDA Memory Errors**:
  - Reduce batch size (`--batch_size`)
  - Enable `--fp16` if disabled
  - Add `--gradient_accumulation_steps`
- **Training Samples Not Generated**:
  - Check file system permissions
  - Verify dataset contains valid samples

## Evaluation Metrics
- **Exact Match**: Percentage of perfectly generated completions
- **Loss Curves**: Monitor training/validation loss in TensorBoard

## License
Apache License 2.0 (Same as original CodeT5 model)