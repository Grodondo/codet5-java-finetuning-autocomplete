import argparse
import logging
import os
import random
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EncoderDecoderCache,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    # Handlers: Where logs go (console, files, etc. - default is console)
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Custom callback to free VRAM and run inference every 10000 steps
class TrainingCallback(TrainerCallback):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = None
        self.dataset = None
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10000 == 0:
            try:
                # Get components from kwargs
                self.model = kwargs.get('model')
                train_dataloader = kwargs.get('train_dataloader')
                self.dataset = train_dataloader.dataset
                print(self.dataset.shape)
                
                if not all([self.model, self.tokenizer, self.dataset]):
                    print("⚠️ Missing components:", 
                          f"Model: {bool(self.model)}, Tokenizer: {bool(self.tokenizer)}, Dataset: {bool(self.dataset)}")
                    return

                # Memory cleanup
                logger.info(f"Step {state.global_step}: Freeing VRAM...")
                torch.cuda.empty_cache()
                gc.collect()

                # Get random tokenized sample
                sample = self.dataset[random.randint(0, len(self.dataset)-1)]
                
                # Decode tokenized input
                input_ids = sample["input_ids"]
                prompt = self.tokenizer.decode(
                    input_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                # Regenerate tokenization (optional but recommended)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding="max_length"
                ).to(self.model.device)

                # Generate output
                outputs = self.model.generate(**inputs)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Write to file
                with open("training_samples.txt", "a") as f:
                    f.write(f"\nStep {state.global_step}:\n")
                    f.write(f"Prompt:\n {prompt}\n")
                    f.write(f"Generated Output:\n {decoded}\n")
                    f.write("-"*50 + "\n")

                logger.info(f"Step {state.global_step}: Sample saved")

            except Exception as e:
                logger.error(f"❌ Callback error: {str(e)}")

def load_model_and_tokenizer(model_name):
    """Initialize model and tokenizer"""
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
    return model, tokenizer


def preprocess_dataset(dataset, split_ratio=0.7):
    """Create prompt-completion pairs from code samples"""
    logger.info(f"Preprocessing dataset with split ratio {split_ratio}")
    
    def process_examples(examples):
        prompts, completions = [], []
        for code in examples["whole_func_string"]:
            lines = code.split('\n')
            split_point = int(len(lines) * split_ratio)
            prompts.append('\n'.join(lines[:split_point]))
            completions.append('\n'.join(lines[split_point:]))
        return {"prompt": prompts, "completion": completions}
    
    return dataset.map(process_examples, batched=True)


def tokenize_data(examples, tokenizer):
    """Tokenize dataset samples
        Structure:
    {
        'input_ids': [prompt tokens],
        'attention_mask': [prompt mask],
        'labels': [completion tokens]
    }
    """
        
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    
    labels = tokenizer(
        examples["completion"],
        max_length=256,
        padding="max_length",
        truncation=True
    ).input_ids
    
    model_inputs["labels"] = labels
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Calculate evaluation metrics"""
    predictions, labels = eval_pred
    # Untokenize the predictions/labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    return {"exact_match": exact_matches / len(decoded_labels)}


def main(args):
    """Main training procedure"""
    
    # Initialize model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load and preprocess dataset
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_language,
        trust_remote_code=True
    )
    
    train_data = preprocess_dataset(dataset["train"])
    valid_data = preprocess_dataset(dataset["validation"])
    
    # Tokenize datasets
    train_dataset = train_data.map(
        lambda ex: tokenize_data(ex, tokenizer),
        batched=True
    )
    val_dataset = valid_data.map(
        lambda ex: tokenize_data(ex, tokenizer),
        batched=True
    )
    
    # Training configuration
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        logging_dir="./logs",
        max_grad_norm=1.0,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
        callbacks=[TrainingCallback(tokenizer=tokenizer)],
    )
    
    # Start training
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    logger.info("Training complete. Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5 Fine-tuning Script")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-small",
                        help="Pretrained model name or path")
    parser.add_argument("--dataset_name", type=str, default="code_search_net",
                        help="Name of the dataset to use")
    parser.add_argument("--dataset_language", type=str, default="java",
                        help="Programming language of the dataset")
    parser.add_argument("--output_dir", type=str, default="./codet5-java",
                        help="Output directory for checkpoints and final model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training and evaluation batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimization")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable mixed precision training")
    parser.set_defaults(fp16=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)