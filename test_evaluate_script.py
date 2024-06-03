from accelerate import Accelerator
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-v1")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba-v1")

# Initialize the accelerator
accelerator = Accelerator()

# Prepare the model, tokenizer, and data for distributed training
model, tokenizer = accelerator.prepare(model, tokenizer)

from transformers import pipeline

# Create a text generation pipeline with your fine-tuned model
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

# The prompt you want to generate text from
prompt = "Here is a sample prompt"

# Generate text 
generated_text = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)[0]['generated_text']

print(generated_text)
