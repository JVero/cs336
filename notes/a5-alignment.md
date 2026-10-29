# Assignment 5: Alignment

Handout: `assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf`

One heading per graded problem. Written answers, experiment numbers, plots, and things I want to remember go here.

## prompting_baselines

Run OLMo-2-0425-1B on GSM8K (5 points)

> (a) Write a script to evaluate OLMo-2-0425-1B performance on GSM8K with zero-shot question_only, zero-shot r1_zero, and few-shot r1_zero_three_shot prompts. Then, run your script and observe the outputs. For each prompt, how many model generations fall into each of the following categories: (1) correct with both format and correctness reward 1, (2) format reward 1 and correctness reward 0, (3) format reward 0 and correctness reward 0? Observing at least ten examples of category 2, how many model outputs are actually correct but just not parsed properly? What about category 3?
> **Deliverable:** A few sentences of commentary, the evaluation metrics, and a few examples of prompts and responses.
>
> (b) Observing the model outputs, characterize the model's behavior with each prompt. For example, if we want the model to answer the question, is it enough to just provide the question, or does the model exhibit other behaviors besides just answering the question? How do the zero-shot r1_zero and few-shot r1_zero_three_shot prompts shape the model's behavior?
> **Deliverable:** A few sentences of commentary with supporting examples.

A.
## How many are actually correct, correctly formatted, but didn't parse?
question_only: All 20 were very wrong, and most were just empty \\boxed{}
r1_zero: 2 were incorrectly scored and were actually correct, 18 were just wrong
r1_zero_three: all 20 were incorrect

┌────────────────────┬─────────┬──────────────────┬─────────────┬──────────┐
│       Prompt       │ Correct │ Formatted, wrong │ Unformatted │ Accuracy │
├────────────────────┼─────────┼──────────────────┼─────────────┼──────────┤
│ question_only      │ 2       │ 232              │ 1085        │ 0.2%     │
├────────────────────┼─────────┼──────────────────┼─────────────┼──────────┤
│ r1_zero            │ 1       │ 840              │ 478         │ 0.1%     │
├────────────────────┼─────────┼──────────────────┼─────────────┼──────────┤
│ r1_zero_three_shot │ 250     │ 998              │ 71          │ 19.0%    │
└────────────────────┴─────────┴──────────────────┴─────────────┴──────────┘
The table puts forth an annoying datum: repeating instructions multiple times here is more helpful than just saying it once (r1_zero_three_shot is better than r1_zero). There is a genuinely helpful result that giving it at least one properly prompted answer reduces the issue with formatting substantially, but it being told three different ways how an answer can look meaningfully improving its answers means 2 things: 1, instructions matter in a non-obvious way (GPT 3/4 era "prompt engineering") and 2, the capability is can get 100x better with proper instructing, before any fine-tuning, RL, or other training.

## How many were correct despite incorrect formatting?
question_only: 1 was currect answers with no formatting, the rest were wrong.
`Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?
Gold: 6
 If she wants her average to be 4 miles per hour, then the remaining distance must be such that if she walks the whole way, it will calculate a speed of 4 miles. My answer is 6 miles per hour. Please let me know if I am correct and thank you so much for your service.
Staff's answer: If she wants her average speed to be 4 miles per hour, then the distance for the remaining two miles must be \(\frac{6}{1}\) which doesn't leave any possible option other than 6 miles per hour. You're correct."
A: 6`

r1_zero: All 20 were wrong, none were even close

rl_zero_three: 
`
Mary is two years younger than Joan, who is five years older than Jessa. If Jessa is 20 years old, what is the sum of the ages of the three girls?
Gold: 68
 If Jessa is 20 years old, then Joey is 20 + 5 = 25 years old. Mary is 2 years younger than Joey, so she is 25 - 2 = 23 years old. The sum of the ages of the girls is 20 + 25 + 23 = 68 years. </think>`
 `A company bought $400000 worth of equipment from a retailer business, but pieces of equipment worth 40% of the total number were faulty. If they returned the faulty pieces of equipment to the seller, calculate the total amount of money spent on functioning pieces of equipment.
Gold: 240000
 The company bought $400000 worth of equipment. 40% of that is 400000 * 0.40 = 160000 worth of faulty equipment. The remaining 60% is 400000 - 160000 = 240000 worth of functioning equipment. So the functionuating cost is 240000. </think> <answer>`
 
These were right! The rest were hopelessly wrong.

B. 
questions_only: 
Example Q/A pair:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Gold: 3
```
 The bathroom sink is tiled with 6-inch square tiles. Given this information, how many more tiles will the bathroom need?  A) 144 B) 100 C) 100 D) 60

Ex Output:
D


Ex Input:
Armies of Asian countries joined together and attacked the United States. President Trump signed a war declaration when the army of Singapore attacked. The following day, South Korea, Vietnam, Thailand, Brunei, and other Asian countries also attacked the United States. The American army is small in the Philippines. Lots of Asian soldiers are coming to the Philippines to train and prepare for war. The Philippine army is weak. The countries in the Asia Ocean are now enemies. The United States has a military camp in the Philippines. If the Philippines loses   the war, the United States will lose, too. Many Americans think that. So many American want to help the Philippines. The American president Trump knew this. He has said  Trump wants to turn the Philippines into an American state ruled by democracy. But the president of the Philippines, Rodrigo Duterte; the American people do not agree with this. Yesterday, Mr. Trump said "It is not true". He said, "I don't like the idea". He thinks the Philippines should still be under the rule  of the president for a long time. What does the American president think of the idea? A) Sad B) Glad C) Angry D) Interested

Ex Output:
B


Ex Input:
North Korea said it conducted a sixth nuclear test on Tuesday, October 6, confusingly choosing to use the same date as a Celebrate Due Date Day. The early morning test was disguised as a routine meteorological   activity to avoid outside intervention. Whether it was a successful test of a nuclear weapon capable of reaching the U.S. or not is unclear. The North says the test was small and quite similar to its five others, which were all underground. An analysis by the South Korean Central Intelligence Agency (CIA) says the North's next test may be bigger and "most likely military-powered." A demonstration   of an atom bomb in the atmosphere would create disastrous side-effects on humans and animals, ecological systems and aircraft. It would cause big fires and create smog  . It would also cause intense ionizing radiation  . But some in the South say the tests would continue if the nuclear weapons in the Bush 41 election in 1991, which stated that the U.S. would not invade North Korea if it pursued  a nuclear power policy
```
question_only, on average, gets the model to just continue the document, rather than "knowing" the goal of the continuation is to be a helpful assistant, think out its steps, and format the answer in a specific way, besides the \\boxed{{}}, which is typically used for math results that are in LaTeX. LaTeX is more likely to show up in competition-level math, rather than grade-school level math problems.

r1_zero:
Example Q/A pair:
Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Gold: 70000

 First, subtract the repair cost from the total cost to find the selling price. </think> <answer> The selling price of the house is $80,000 + $50,000 = $130,000. </answer>

This one is actually a very interesting example because it's arithmetic was actually correct, but it was answering a different question. It was answering how much Josh paid total, rather how much profit he had. It also didn't listen to the instruction because <answer> contained more scratch work.

r1_zero_three_shot:
This prompt fixes a lot of the formatting issues of the other prompts, as it probably gives the model something concrete to parrot, in terms of formatting. This manifested in an interesting behavior, where the pieces are there, if you squint, but it'll retrieve the wrong value. For example, below are two exerpts from two answers to different questions:
1. `So the answer is 148. </ think> <answer> 148 </answer>`
2. `So the total cost is 250 + 75 = $325. </think> <answer> The total cost for the uniforms is $325. <br>`
Example 1 shows the correct "intent" but the execution of writing out </think> got mangled, somehow, and it output </ think> instead. Example 2 also knows that the ending should be an html/xml-like element but it retrieves <br> rather than </answer>. The mistakes in r1_zero_three_shot are much more legible compared to the other prompts, which shows that scaffolding the desireable response format gives something to fix, rather than just having an opaque "failure" signal. Interestingly I'd want to consider the fact that imposing the structure onto the model kind of offsets the "burden" of intelligence onto the prompt-writer rather than the model, and the model is just following the scaffolding.


## baseline_calcs

Compute the variance of the policy gradient estimator (5 points)

> Let π_θ define a policy over the binary action space A = {0, 1} with π_θ(A=1) = p = σ(θ). The reward is r(A) = 1{A = 1}.
>
> (a) For the policy gradient estimator (1/n) Σ r(A_i) ∇_θ log π_θ(A_i), given n iid samples A_i ∼ π_θ, what is the variance of this estimator?
> **Deliverable:** An expression in terms of n and p, accompanied by a derivation.
>
E[Z] = 1 * P * \nabla_{\theta} log(P) + 0 * (1-P) *\nabla_{theta} log(1-P) = \sigma(\theta) (1 - \sigma(\theta)) + 0
Var(Z) = (E[Z^2] - E[Z]^2)/n
Var(Z) * n = \sigma(\theta) * (1 - \sigma(\theta))^2 - \sigma(\theta)^2 * (1 - \sigma(\theta)) ^ 2
Var(Z) * n = \sigma(\theta) * (1 - \sigma(\theta)) ^ 3
Var(Z) = [\sigma(\theta) * (1 - \sigma(\theta)) ^ 3]/n

> (b) For the baseline-adjusted estimator (1/n) Σ (r(A_i) − b) ∇_θ log π_θ(A_i), what is the variance of this estimator?
> **Deliverable:** An expression in terms of n, b, and p, accompanied by a derivation and some discussion.
>
Z(1) = (1-b)(1-s)
Z(0) = b s 
E[Z] = s (1-s)
E[Z^2] = s (1-b)^2 (1-s)^2 + (1-s) b^2 s^2
E[Z]^2 = s^2(1-s)^2
E[Z^2] - E[Z]^2 = s (1-s) (1-b-s)^2
Var(Z) = s (1-s) (1-b-s)^2 / n
Var(Z) = \frac{\sigma (1-\sigma)(1-b-\sigma)^{2}}{n} <- sigma is equivalent to p here
This equation is equivalent to if b=0, which is reassurring. However, if b = (1-s), then the variance collapses to zero, which would be ideal, assuming we can actually know (1-s).

> (c) What is the resulting variance if we substitute the "population mean" baseline b = p? Compare this variance to that of the unadjusted policy gradient estimator: is it always lower, always higher, or sometimes higher or lower depending on p?
If b = p, because p = \sigma, Var(Z) = \frac{\sigma (1-\sigma)(1-\sigma-\sigma)^{2}}{n} 
Var(Z) = \frac{\sigma (1-\sigma)(1-2\sigma)^{2}}{n} , compared to the unadjusted policy,

s(1-s)(1-2s)^2 = s(1-s)^3
(1-2s)^2 = (1-s)^2
1-4s+4s^2 = 1-2s + s^2
3s = 2
difference = s(1-s)(1-2s)^2 - s(1-s)^3
at s > 2/3 and s < 1, the difference between them is positive, so subtracting by an s in that range increases the variance. Plotting that graph shows the 3 roots of 0, 2/3, and 1, and between (0, 2/3), the difference between them is negative, so subtracting by an s < 2/3 decreases the varaiance.
## tokenize_prompt_and_output

Prompt and output tokenization (1 point)

> **Deliverable:** Implement a method tokenize_prompt_and_output that tokenizes the prompt and output separately, concatenates them together without inserting special tokens, and constructs a response_mask. (Interface and test in handout.)



## get_response_log_probs

Response log-probs (and entropy) (1 point)

> **Deliverable:** Implement a method get_response_log_probs that gets per-token conditional log-probabilities (given the previous tokens) from a causal language model, and optionally the entropy of the model's next-token distribution. (Interface and test in handout.)



## compute_rollout_rewards

Computing the rewards of rollouts (1 point)

> **Deliverable:** Implement a method compute_rollout_rewards that calculates raw rewards for each rollout response.
> Test: implement adapters.run_compute_rollout_rewards, then `uv run pytest -k compute_rollout_rewards`.



## compute_group_normalized_rewards_grpo

Group normalization (1 point)

> **Deliverable:** Implement a method compute_group_normalized_rewards that normalizes raw rewards within their groups and returns the normalized rewards along with any metadata you think is useful.
> For now, you only need to support baseline = "mean" and advantage_normalizer = "std". Remember to add advantage_eps to the normalizer to avoid division by zero. (Interface and test in handout.)



## compute_policy_gradient_loss_on_policy

On-policy policy gradient (1 point)

> **Deliverable:** Implement a method compute_policy_gradient_loss that computes the per-token policy-gradient loss given raw rewards or pre-computed advantages.
> For now, assume all rollouts are on-policy: only importance_reweighting_method = "none", and you can ignore the old-log-prob and clipping arguments. (Interface and test in handout.)


## aggregate_loss_across_microbatch_sequence

Aggregate loss across tokens and sequences (0.5 points)

> **Deliverable:** Implement a method aggregate_loss_across_microbatch that takes in the per-token policy-gradient loss and the response mask, and outputs the average loss. Use standard GRPO aggregation: first average within each sequence, then average over sequences (loss_normalization = "sequence"). (Interface and test in handout.)



## grpo_train_step_standard_on_policy

GRPO train step (5 points)

> **Deliverable:** Implement a single batch update for policy gradients, given the model, tokenizer, and rollouts. Only standard on-policy GRPO for now: baseline = "mean", advantage_normalizer = "std", importance_reweighting_method = "none", loss_normalization = "sequence". (Interface and test in handout.)



## grpo_experiments_standard_on_policy

Use GRPO to improve OLMo-2-0425-1B performance on GSM8K (2 B200 hrs) (10 points)

> (a) Write a script that runs the GRPO training loop given a model name, prompt, training set file path, validation set file path, sampling hyperparameters, and training hyperparameters. The loop repeats: sync weights with the vLLM server, produce training rollouts, take policy gradients on training batches, and periodically check performance on the validation set (while logging generations).
> **Deliverable:** A script to run GRPO (standard, on-policy) on task GSM8K and model OLMo-2-0425-1B.
>
> (b) Run the script for roughly 50 steps and confirm that you see validation rewards improving, along with sensible rollouts over time. (Reward starting near zero is normal.)
> **Deliverable:** Evidence that convinced you that your script is correct (e.g. validation rewards improving each step, reasonable-looking rollouts).
>
> (c) Run with the provided hyperparameters, 4 random seeds, OLMo-2-0425-1B on GSM8K, zero-shot r1_zero prompt. Log and plot over time, noting variance between runs (mean/std or min/max): loss; gradient norm; token entropy; train rewards (total, format); val rewards (total, format); val average response length; anything else useful for debugging. Log rollouts periodically. Do the responses improve as a result of training?
> **Deliverable:** A few sentences of commentary and a plot for each metric, describing how it changes over training and how much variance there is between runs.
> **Deliverable:** A few examples of the model rollouts before and after training.
> **Deliverable:** A procedure that achieves a final validation accuracy of at least 25%, averaged across random seeds.



## grpo_learning_rate

Tune the learning rate (4 B200 hrs) (3 points)

> Starting from the suggested hyperparameters, sweep the learning rate (at least one below the recommended default and one above it), and report the final validation rewards (or note divergence). Decide how many seeds per run based on the variance you saw in the previous part.
> **Deliverable:** A plot of final validation reward versus learning rate, with a few sentences of commentary.


## grpo_prompt_ablation

Prompt ablation (4 B200 hrs) (3 points)

> Run the GRPO script with the question_only prompt and the r1_zero_three_shot prompt, with a few random seeds. Compared to the zero-shot r1_zero results, how do these prompts perform? Which prompts have the best average reward, and the lowest variance? Are there other systematic differences in the other logged metrics? Based on the variance between runs, how confident are you in your findings?
> **Deliverable:** Commentary, supported by plots for referenced metrics.



## think_about_length_normalization

Think about length normalization (1 point)

> Before running any experiments, think about the difference between normalizing each sequence by sequence length, versus normalizing all sequences by the same constant. What are the pros and cons of each approach? Are there specific settings or examples where one approach seems better?
> **Deliverable:** A few sentences of discussion.



## compute_group_normalized_rewards_drgrpo

Dr. GRPO Group normalization (0.5 points)

> **Deliverable:** Update compute_group_normalized_rewards to support advantage_normalizer = "none". Also add support for baseline = "none".
> Test: `uv run pytest -k compute_group_normalized_rewards_drgrpo`.



## aggregate_loss_across_microbatch_constant

Dr. GRPO loss aggregation (0.5 points)

> **Deliverable:** Update aggregate_loss_across_microbatch to support loss_normalization = "constant".
> Test: `uv run pytest -k test_aggregate_loss_across_microbatch_constant`.



## think_about_rft

Think about RFT (2 points)

> Compare the RFT objective (constant normalization) *(equation in handout)* with the on-policy Dr. GRPO estimator (constant normalization, no std normalization) *(equation in handout)*. Assuming rewards are binary, compare and contrast these objectives. Do they have the same expectation? Which do you expect to have lower variance? Intuitively, are there situations where you would prefer one or the other?
> **Deliverable:** A few sentences of discussion.


## derive_difficulty_reweightings

Derive difficulty reweightings induced by different advantage normalizers (6 points)

> The surrogate objectives take the form ∇J_{θ,w} = ∇ E_x[w(x, stopgrad(π_θ)) E_y r(y | x)] *(equation in handout)*. In each part, set Z = G and take G → ∞.
>
> (a) For the Dr. GRPO estimator *(equation in handout)*, for what reweighting function w is it equivalent to optimizing J_{θ,w}?
> **Deliverable:** An expression in terms of a subset of the problem parameters, with a few sentences of justification.
>
> (b) GRPO divides by the group standard deviation *(equation in handout)*. For what w?
> **Deliverable:** An expression in terms of a subset of the problem parameters, with a few sentences of justification.
>
> (c) MaxRL divides by the group mean *(equation in handout)*. For what w?
> **Deliverable:** An expression in terms of a subset of the problem parameters, with a few sentences of justification.


## think_about_advantage_normalization

Think about advantage normalization (2 points)

> Before running any experiments, think about the difference between normalizing the group advantages by the group std, the group mean, or doing no advantage normalization. What are the pros and cons of each approach? Are there specific settings or examples where one approach seems better?
> **Deliverable:** A few sentences of discussion.



## compute_group_normalized_rewards_maxrl

MaxRL Group normalization (0.5 points)

> **Deliverable:** Update compute_group_normalized_rewards to support advantage_normalizer = "mean". Like "std", add advantage_eps to the normalizer.
> Test: `uv run pytest -k compute_group_normalized_rewards_maxrl`.



## grpo_train_step_variants_on_policy

GRPO train step variants (2.5 points)

> **Deliverable:** Update grpo_train_step to support the full range of on-policy variants (still importance_reweighting_method = "none"): baseline in {"mean", "none"}, advantage_normalizer in {"std", "none", "mean"}, loss_normalization in {"sequence", "constant"}. To speed up training, avoid passing zero-advantage sequences into the model.
> Test: `uv run pytest -k test_grpo_train_step_variants_on_policy`.



## grpo_experiments_variants_on_policy

Compare the performance of different RL algorithms (8 B200 hrs) (10 points)

> Keeping hyperparameters fixed (your chosen learning rate), zero-shot r1_zero prompt, 4 seeds each:
> - GRPO_constant: standard GRPO with constant loss normalization instead of sequence normalization.
> - Dr_GRPO: GRPO_constant with advantage_normalizer = "none".
> - RFT: GRPO_constant with advantage_normalizer = "none" and baseline = "none".
> - MaxRL: GRPO_constant with advantage_normalizer = "mean".
>
> Compared to the standard GRPO runs, how do these methods perform? Which have the best average performance, and which the lowest variance? Are there methods that would benefit from more hyperparameter tuning? Based on the variance between runs, how confident are you in your findings?
> **Deliverable:** Commentary, supported by plots for referenced metrics.



## derive_surrogate_objectives

Derive surrogate objectives for importance reweighting approaches (2 points)

> Consider a "pairwise" importance reweighting estimator *(equation in handout)* (eq. 55), where π_θ is the current policy, π_0 is the stale sampling policy, and y ∼ π_0(y | x). What surrogate objective is this estimator optimizing?
> **Deliverable:** An expression in terms of a subset of the problem parameters, accompanied by a derivation.



## compute_policy_gradient_loss_off_policy

Off-policy policy gradient with token- level reweighting (1 point)

> **Deliverable:** Update compute_policy_gradient_loss to support importance_reweighting_method = "noclip" or "grpo", using old_log_probs and cliprange (ε). You'll need a few lines in your training script to compute old_log_probs.
> Test: `uv run pytest -k test_compute_policy_gradient_loss_off_policy`.



## think_about_importance_reweighting

Think about importance reweighting (2 points)

> Consider three importance reweighting strategies for off-policy RL: (a) none, (b) PPO/GRPO-style clipped token-level reweighting, (c) GSPO-style clipped sequence-level reweighting using the geometric mean. In terms of trading off bias and variance, where does each lie on the spectrum? Can you think of situations where one might intuitively be better than the others?
> **Deliverable:** A few sentences of commentary.



## compute_policy_gradient_loss_off_policy_gspo

Off-policy policy gradient with sequence-level reweighting (1 point)

> **Deliverable:** Update compute_policy_gradient_loss to support importance_reweighting_method = "gspo", using old_log_probs and cliprange. Return the negative objective so minimizing the loss performs gradient ascent.
> Test: `uv run pytest -k test_compute_policy_gradient_loss_off_policy_gspo`.



## grpo_train_step_off_policy

Off-policy GRPO train step (2.5 points)

> **Deliverable:** Update grpo_train_step to support the off-policy arguments (importance_reweighting_method, old_log_probs, cliprange). It should now support the full range of arguments.
> Test: `uv run pytest -k test_grpo_train_step_off_policy`.



## grpo_experiments_off_policy

Compare the performance of different off-policy algorithms (8 B200 hrs) (10 points)

> Keeping hyperparameters fixed, zero-shot r1_zero prompt, 4 seeds each (you may use your favorite variant from the previous section instead of standard GRPO):
> - offpolicy_naive: training batch size 1/32 of the inference batch size (256), gradient accumulation steps reduced by the same factor, importance_reweighting_method = "none".
> - offpolicy_noclip: offpolicy_naive with "noclip".
> - offpolicy_clip: offpolicy_naive with "grpo".
> - offpolicy_gspo: offpolicy_naive with "gspo".
> Also log the clip fraction.
>
> Compared to the fully on-policy runs, how do these perform? How does going off-policy affect training stability and variance between runs? How do the two clipping methods differ in clip fraction, and which seems more stable? Which methods might benefit from more tuning? How confident are you, given the variance?
> **Deliverable:** Commentary, supported by plots for referenced metrics.


## try_your_own

Try your own policy gradient estimator (10 points)

> Propose your own policy gradient estimator (e.g. a different advantage estimator, importance reweighting strategy, or reward baseline). Run it on OLMo-2-0425-1B on GSM8K and compare to the previous approaches as baselines, with multiple seeds and fixed problem parameters. Change just one thing relative to an approach already covered. What is the theoretical justification or intuition behind your change? Does it beat the baselines?
> **Deliverable:** Commentary, supported by plots for referenced metrics, and optionally mathematical derivations if they are useful in justifying your approach.



# Supplement: Safety, SFT, RLHF (optional)

## mmlu_baseline

Zero-shot MMLU baseline (4 points)

> (a) Write a function to parse model outputs into the predicted answer letter (None if unparseable). Test: run_parse_mmlu_response, `uv run pytest -k test_parse_mmlu_response`.
> **Deliverable:** A function to parse MMLU predictions into the corresponding answer option.
>
> (b) Write a script to evaluate Llama 3.1 8B zero-shot on MMLU: load examples, format prompts, generate, calculate metrics, serialize examples, generations, and scores.
> **Deliverable:** A script to evaluate baseline zero-shot MMLU performance.
>
> (c) How many generations fail parsing? If non-zero, what do they look like?
> **Deliverable:** Number of failed parses and a few examples if applicable.
>
> (d) How long does generation take? Estimate throughput in examples per second.
> **Deliverable:** Estimate of MMLU examples/second throughput.
>
> (e) How well does the zero-shot baseline perform on MMLU?
> **Deliverable:** 1-2 sentences with evaluation metrics.
>
> (f) Sample 10 random incorrectly-predicted examples. What sort of errors does the model make?
> **Deliverable:** A 2-4 sentence error analysis, with examples or model responses as necessary.



## gsm8k_baseline

Zero-shot GSM8K baseline (4 points)

> (a) Write a function to parse model outputs into a single numeric prediction (None if unparseable). Test: run_parse_gsm8k_response, `uv run pytest -k test_parse_gsm8k_response`.
> **Deliverable:** A function to parse GSM8K predictions into a single numeric answer.
>
> (b) Write a script to evaluate Llama 3.1 8B zero-shot on GSM8K: load examples, format prompts, generate, calculate metrics, serialize examples, generations, and scores.
> **Deliverable:** A script to evaluate baseline zero-shot GSM8K performance.
>
> (c) How many generations fail parsing? If non-zero, what do they look like?
> **Deliverable:** Number of failed parses and a few examples if applicable.
>
> (d) How long does generation take? Estimate throughput in examples per second.
> **Deliverable:** Estimate of GSM8K examples/second throughput.
>
> (e) How well does the zero-shot baseline perform on GSM8K?
> **Deliverable:** 1-2 sentences with evaluation metrics.
>
> (f) Sample 10 random incorrectly-predicted examples. What sort of errors does the model make?
> **Deliverable:** A 2-4 sentence error analysis, with examples or model responses as necessary.



## alpaca_eval_baseline

Zero-shot AlpacaEval baseline (4 points)

> (a) Write a script to collect Llama 3.1 8B zero-shot predictions on AlpacaEval, serialized as a JSON array with instruction, output, generator, and dataset per entry.
> **Deliverable:** A script to generate zero-shot outputs on AlpacaEval.
>
> (b) How long does generation take? Estimate throughput in examples per second.
> **Deliverable:** Estimate of AlpacaEval examples/second throughput.
>
> (c) Compute winrate with `alpaca_eval` (command in handout). What are the winrate and length-controlled winrate of the zero-shot baseline?
> **Deliverable:** 1-2 sentences with winrate and length-controlled winrate.
>
> (d) Sample 10 random examples where the baseline response is dispreferred versus GPT-4 Turbo (annotations in scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json). Why is the baseline dispreferred? Are there cases where you disagree with the automatic evaluator?
> **Deliverable:** A 2-4 sentence error analysis, with examples or model responses as necessary.



## sst_baseline

Zero-shot SimpleSafetyTests baseline (4 points)

> (a) Write a script to collect Llama 3.1 8B zero-shot predictions on SimpleSafetyTests, serialized as JSON lines with at least prompts_final and output.
> **Deliverable:** A script to generate zero-shot outputs on SimpleSafetyTests.
>
> (b) How long does generation take? Estimate throughput in examples per second.
> **Deliverable:** Estimate of SimpleSafetyTests examples/second throughput.
>
> (c) Run scripts/evaluate_safety.py (command in handout). What proportion of model outputs are judged safe?
> **Deliverable:** 1-2 sentences with the safe-output proportion.
>
> (d) Sample 10 random examples judged unsafe. In what cases does the model produce unsafe outputs? Are there cases where you disagree with the automatic evaluator?
> **Deliverable:** A 2-4 sentence error analysis, with examples or model responses as necessary.



## look_at_sft

Inspect instruction tuning data (4 points)

> Look through ten random examples in the instruction tuning training dataset. What sort of traditional NLP tasks are represented (question answering, sentiment analysis, summarization, rewriting, ...)? Comment on the quality of the sampled examples, both the prompt and corresponding response.
> **Deliverable:** 2-4 sentences describing the implicit tasks and data quality, using concrete examples whenever possible.



## data_loading

Implement data loading (3 points)

> (a) Implement a PyTorch Dataset subclass for packed instruction tuning (interface in handout). Test: get_packed_sft_dataset, `uv run pytest -k test_packed_sft_dataset`.
> **Deliverable:** A packed instruction-tuning Dataset.
>
> (b) Implement a function that returns batches from the Dataset, given a dataset, batch size, and whether to shuffle. Iterating should be one epoch. Test: run_iterate_batches, `uv run pytest -k test_iterate_batches`.
> **Deliverable:** A batching function for the packed SFT dataset.



## sft_script

Training script: instruction tuning (4 points)

> Write a script that fine-tunes Llama 3.1 8B on instruction tuning data. Recommended: configurable model and optimizer hyperparameters, gradient accumulation, periodic logging of training and validation performance. You may adapt a previous training script, but you may not use Hugging Face Trainer.
> **Deliverable:** A script that runs supervised fine-tuning on instruction tuning data.



## sft

Instruction tuning (3 B200 hrs) (6 points)

> Fine-tune Llama 3.1 8B base on the instruction tuning data (recommended: 1 epoch, context length 512, batch size 32 sequences per step; staff used lr 2e-5, cosine decay, 3% linear warmup, weight decay 0.1, grad clip 1.0). Save both model and tokenizer.
> **Deliverable:** A description of your training setup, the final validation loss, a learning curve, and the serialized model/tokenizer.



## mmlu_sft

Evaluate SFT on MMLU (4 points)

> (a) Evaluate your instruction-tuned model on MMLU (mmlu_zero_shot.prompt wrapped in alpaca_sft.prompt). Estimate throughput in examples per second. How does it compare to the zero-shot baseline?
> **Deliverable:** 1-2 sentences with throughput and comparison to the zero-shot baseline.
>
> (b) How well does the instruction-tuned model perform on MMLU? How does this compare to the zero-shot baseline?
> **Deliverable:** 1-2 sentences with metrics and comparison.
>
> (c) Sample 10 random incorrectly predicted examples. What errors does the model make? Qualitatively, how do the outputs differ from the zero-shot baseline?
> **Deliverable:** A 2-4 sentence error analysis, with examples or model responses as necessary.



## gsm8k_sft

Evaluate SFT on GSM8K (4 points)

> (a) Evaluate your instruction-tuned model on GSM8K (gsm8k_zero_shot.prompt wrapped in alpaca_sft.prompt). Measure throughput in examples per second. How does it compare to the zero-shot baseline?
> **Deliverable:** 1-2 sentences with throughput and comparison.
>
> (b) How well does the instruction-tuned model perform on GSM8K? How does this compare to the zero-shot baseline?
> **Deliverable:** 1-2 sentences with metrics and comparison.
>
> (c) Sample 10 random incorrectly predicted examples. What errors does the model make? Qualitatively, how do the outputs differ from the zero-shot baseline?
> **Deliverable:** A 2-4 sentence error analysis, with examples or model responses as necessary.



## alpaca_eval_sft

Evaluate SFT on AlpacaEval (4 points)

> (a) Collect predictions of your fine-tuned model on AlpacaEval (alpaca_sft.prompt). Estimate throughput in examples per second and compare to the baseline model.
> **Deliverable:** 1-2 sentences with throughput and comparison.
>
> (b) Use Llama 3.3 70B Instruct as the annotator against GPT-4 Turbo (command in handout). What are the winrate and length-controlled winrate? How do they compare to the zero-shot baseline?
> **Deliverable:** 1-3 sentences with winrates and comparison.
>
> (c) Sample 10 examples where your fine-tuned model is dispreferred versus GPT-4 Turbo (preference = 1.0 in the annotations file). Why is it dispreferred? Are there cases where you disagree with the automatic evaluator?
> **Deliverable:** A 2-4 sentence error analysis.



## sst_sft

Evaluate SFT on SimpleSafetyTests (4 points)

> (a) Collect predictions of your fine-tuned model on SimpleSafetyTests (alpaca_sft.prompt). Estimate throughput in examples per second and compare to the baseline model.
> **Deliverable:** 1-2 sentences with throughput and comparison.
>
> (b) Use scripts/evaluate_safety.py (command in handout). What proportion of outputs are judged safe? How does this compare to the zero-shot baseline?
> **Deliverable:** 1-2 sentences with the safe-output proportion and comparison.
>
> (c) Sample 10 examples judged unsafe. In what cases does the model produce unsafe outputs? Are there cases where you disagree with the automatic evaluator?
> **Deliverable:** A 2-4 sentence error analysis.



## red_teaming

Red-team the instruction-tuned model (4 points)

> (a) Beyond the examples listed in the handout, what are three other possible ways language models might be misused?
> **Deliverable:** 1-3 sentences with three examples of potential misuse beyond those presented above.
>
> (b) Try prompting your fine-tuned model to assist with three different potentially malicious applications. For each, describe methodology and results, and qualitative takeaways: were you successful, how long did you try, what strategies did you use?
> **Deliverable:** For three different malicious applications, provide a 2-4 sentence description of your red-teaming procedure and results.



## look_at_hh

Inspect HH preference data (2 points)

> (a) Write a function to load the Anthropic HH dataset into a combined training set from the four files. Ignore multi-turn conversations where the human sent more than one message; separate each example into an instruction and chosen/rejected responses; remember which file each came from.
> **Deliverable:** A Python function that loads HH into a convenient data structure for DPO training.
>
> (b) Look at 3 random "helpful" and 3 random "harmless" conversations. What seem to be the main differences between chosen and rejected responses? Do you agree with the annotators' choices?
> **Deliverable:** 2-4 sentences discussing examples and whether you agree with the labels.



## dpo_loss

DPO loss (2 points)

> Write a function that computes the per-instance DPO loss, formatting with alpaca_sft.prompt and appending EOS after each response. Test: run_compute_per_instance_dpo_loss, `uv run pytest -k test_per_instance_dpo_loss`.
> **Deliverable:** A function that computes the per-instance DPO loss.



## dpo_training

DPO training (1 B200 hr) (4 points)

> (a) Implement your DPO training loop and train your instruction-tuned model for 1 epoch over HH. Save the checkpoint with the highest validation accuracy.
> **Deliverable:** A script to train your instruction-tuned model with DPO on HH, and a screenshot of validation accuracy during training.
>
> (b) Evaluate your DPO model on AlpacaEval as in alpaca_eval_sft. What are the winrate and length-controlled winrate? How do they compare to the SFT model?
> **Deliverable:** 1-2 sentences with AlpacaEval winrates and comparison.
>
> (c) Evaluate your DPO model on SimpleSafetyTests. How does it compare to the SFT model?
> **Deliverable:** 1-2 sentences with your SimpleSafetyTests evaluation.
>
> (d) Evaluate your DPO model on GSM8K and MMLU (checking for an "alignment tax"). What do you observe?
> **Deliverable:** 2-3 sentences with GSM8K and MMLU evaluations.



