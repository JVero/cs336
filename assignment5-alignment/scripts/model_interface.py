import transformers

olmo = transformers.AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

def prompt(text: str, max_new_tokens=100) -> str:
    inputs = tokenizer(text, return_tensors='pt')
    response_tokens = olmo.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95) # type: ignore
    response = tokenizer.decode(response_tokens)
    return response
    
if __name__ == "__main__":
    prompt("hello")