from cs336_alignment.vllm_utils import VLLMServer, VLLMCompletion
import json
from copy import deepcopy
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

from collections import Counter

server: VLLMServer = None # type: ignore

sampling_params = {
    'temperature': 1.0,
    'top-p': 1.0,
    'max_tokens': 512,
    'stop': '</answer>',
    'include_stop_str_in_output': True,
    'seed': 0,
    'n': 1
}

payload = {
    # "model": model_id,
    # "prompt": prompt_batch,
    "temperature": sampling_params["temperature"],
    "max_tokens": sampling_params["max_tokens"],
    "n": sampling_params["n"],
    "seed": sampling_params["seed"],
    "return_token_ids": True,
    "include_stop_str_in_output": True
}

def init():
    global server
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
    # server = VLLMServer(model_id="allenai/OLMo-2-1124-7B", gpu=0)
    server.start()

with open("cs336_alignment/prompts/question_only.prompt", "r") as f:
    Q_PROMPT = f.read()
    
with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
    R1_PROMPT = f.read()

with open("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt", "r") as f:
    R1_3_PROMPT = f.read()

def format_prompts(prompts: list[str], TEMPLATE: str) -> list[str]:
    return [TEMPLATE.format(question=prompt) for prompt in prompts]

def question_only(prompts: list[str], sampling_params: dict=sampling_params) -> list[VLLMCompletion]:
    sampling_params = deepcopy(sampling_params)
    del sampling_params["stop"]
    del sampling_params["include_stop_str_in_output"]
    prompts = format_prompts(prompts, Q_PROMPT)
    return server.generate_completions(prompts, sampling_params)

def r1_zero(prompts: list[str], sampling_params: dict=sampling_params) -> list[VLLMCompletion]:
    prompts = format_prompts(prompts, R1_PROMPT)
    return server.generate_completions(prompts, sampling_params)


def r1_zero_three_shot(prompts: list[str], sampling_params: dict=sampling_params) -> list[VLLMCompletion]:
    prompts = format_prompts(prompts, R1_3_PROMPT)
    return server.generate_completions(prompts, sampling_params)

def load_data(fname="data/gsm8k/test.jsonl", only_answers=True, num_samples=1319) -> list[dict]:
    questions = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            q = json.loads(line)
            assert "question" in q.keys() and "answer" in q.keys()
            if only_answers:
                q["answer"] = q["answer"].split("####")[1].strip()
            questions.append(q)
    return questions[:num_samples]

def evaluate(eval_fn, responses: list[VLLMCompletion], answers: list[str]):
    results = []
    total_results = {"Correct": 0, "Correct Formatting": 0, "Unformatted": 0}
    for response, answer in zip(responses, answers):
        results.append(eval_fn(response.text, answer))
        """                "format_reward": 1.0,
                "answer_reward": 0.0,
                "reward": 0.0"""
        
        if results[-1]["reward"] == 1.0: # correct answer
            total_results["Correct"] += 1
        elif results[-1]["format_reward"] == 1.0: # correct formatting
            total_results["Correct Formatting"] += 1
        else:
            total_results["Unformatted"] += 1
    return results, total_results

def grade_questions(num_samples=10):
    data = load_data(num_samples=num_samples)
    questions = [line["question"] for line in data]
    answers = [line["answer"] for line in data]
    qo_answers = question_only(questions, sampling_params)
    eval_qo, qo_dict = evaluate(question_only_reward_fn, qo_answers, answers)
    r1_answers = r1_zero(questions, sampling_params)
    eval_r1, r1_dict = evaluate(r1_zero_reward_fn, r1_answers, answers)
    r1_zero_answers = r1_zero_three_shot(questions, sampling_params)
    eval_r1_zero, r1_zero_dict = evaluate(r1_zero_reward_fn, r1_zero_answers, answers)
    
    results_as_dict = [
        {"name": "question_only", "answers": [answer.text for answer in qo_answers], **qo_dict},
        {"name": "r1_zero", "answers": [answer.text for answer in r1_answers], **r1_dict}, 
        {"name": "r1_zero_three_shot", "answers": [answer.text for answer in r1_zero_answers], **r1_zero_dict}]
    
    data = {
        "data": data,
        "results": results_as_dict
    }
    with open("results.json", "w") as f:
        json.dump(data, f)
    
    

if __name__ == "__main__":
    init()    
    grade_questions(num_samples=1319)
