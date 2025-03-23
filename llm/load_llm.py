from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "mistralai/Mistral-7B-v0.1"



# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token="hf_token")

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
prompt = "What are the benefits of using AI in business?"
output = generator(prompt, max_length=100, do_sample=True)

# Print result
print(output[0]['generated_text'])
