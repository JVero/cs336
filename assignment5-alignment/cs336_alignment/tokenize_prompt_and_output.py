import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device=="cuda" else "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    batch_size = len(prompt_strs)
    prompt_and_output_lens: list[int] = []

    toks = []
    prompt_lens = []
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt = tokenizer.encode(prompt_str)
        output = tokenizer.encode(output_str)
        prompt_lens.append(len(prompt))
        prompt.extend(output)
        toks.append(prompt)
        prompt_and_output_lens.append(len(prompt))
    output = {}
    output["input_ids"] = torch.zeros(batch_size, max(prompt_and_output_lens))    
    output["labels"] = torch.zeros(batch_size, max(prompt_and_output_lens))
    output["response_mask"] = torch.zeros(batch_size, max(prompt_and_output_lens))    
    for i, tokens in enumerate(toks):
        output["input_ids"][i, :prompt_and_output_lens[i]] = torch.Tensor(tokens)
        output["labels"][i, :prompt_and_output_lens[i]] = torch.Tensor(tokens)
        output["response_mask"][i, prompt_lens[i]:(prompt_and_output_lens[i])] = 1
    output["input_ids"] = output["input_ids"][:, :-1]
    output["labels"] = output["labels"][:, 1:]
    output["response_mask"] = output["response_mask"][:, 1:]
    return output

if __name__ == "__main__":
    output_dir = "."
    model, tokenizer = get_model_and_tokenizer("allenai/OLMo-2-0425-1B", "mps")
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)