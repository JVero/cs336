import json
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
if __name__ == "__main__":
    results_fname = "./results_1b.json"
    new_results = {}
    with open(results_fname, "r") as f:
        results = json.load(f)
    data = results["data"]
    results_qo = results["results"][0]["answers"]
    results_r1_0 = results["results"][1]["answers"]
    results_r1_3 = results["results"][2]["answers"]
    
    outputs = []
    for sample, result_qo, result_r1_0, result_r1_3 in zip(data, results_qo, results_r1_0, results_r1_3):
        grade_qo = question_only_reward_fn(result_qo, sample["answer"])
        grade_r1_0 = r1_zero_reward_fn(result_r1_0, sample["answer"])
        grade_r1_3 = r1_zero_reward_fn(result_r1_3, sample["answer"])
        
        outputs.append({
            "data": sample,
            "question_only": [result_qo, grade_qo],
            "r1_zero": [result_r1_0, grade_r1_0],
            "r1_zero_three_shot": [result_r1_3, grade_r1_3]
        })
        
    with open("results_1B.jsonl", "w") as f:
        for output in outputs:
            f.write(json.dumps(output)+"\n")
        