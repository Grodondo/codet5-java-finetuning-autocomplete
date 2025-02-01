# inference.py
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(model_path, model_name):
    """Load model and try to extract tokenizer from model config"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
        
    # Load model first
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Try to load tokenizer from model directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"Tokenizer not found in {model_path}, falling back to {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    return model, tokenizer

def generate_completion(model, tokenizer, code_prompt, max_length=256):
    """Generate code completion for a given prompt"""
    inputs = tokenizer(
        code_prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    outputs = model.generate(
        inputs.input_ids.to(model.device),
        attention_mask=inputs.attention_mask.to(model.device),
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate code completions')
    parser.add_argument('--model_path', type=str, default='./codet5-java/final_model',
                       help='Path to trained model')
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5-small',
                       help='(Optional) Name of the model, in case the model_path does not contain a tokenizer')
    parser.add_argument('--input_code', type=str,
                       help='Code snippet to complete')
    parser.add_argument('--input_code_file', type=str,
                       help='(Optional) File with the code snippet to complete, overrides input_code')
    parser.add_argument('--max_length', type=int, default=256,
                       help='(Optional) Maximum length of generated code')
    
    args = parser.parse_args()
    
    # Ensure an input is provided
    if not (args.input_code or args.input_code_file):
        parser.error("One of --input_code or --input_code_file must be provided.")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Generate completion
    if args.input_code_file:
        with open(args.input_code_file, "r") as f:
            # If a file is provided, read the code from the file
            args.input_code = f.read()
    completion = generate_completion(model, tokenizer, args.input_code, args.max_length)
    
    # Print results
    print("\nInput Code:")
    print(args.input_code)
    print("\nGenerated Completion:")
    print(completion)