# Assignment 5: Alignment

Handout: `assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf`

One heading per graded problem. Written answers, experiment numbers, plots, and things I want to remember go here.

## prompting_baselines

Run OLMo-2-0425-1B on GSM8K (5 points)
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


## tokenize_prompt_and_output

Prompt and output tokenization (1 point)


## get_response_log_probs

Response log-probs (and entropy) (1 point)


## compute_rollout_rewards

Computing the rewards of rollouts (1 point)


## compute_group_normalized_rewards_grpo

Group normalization (1 point)


## aggregate_loss_across_microbatch_sequence

Aggregate loss across tokens and sequences (0.5 points)


## grpo_train_step_standard_on_policy

GRPO train step (5 points)


## grpo_experiments_standard_on_policy

Use GRPO to improve OLMo-2-0425-1B performance on GSM8K (2 B200 hrs) (10 points)


## grpo_prompt_ablation

Prompt ablation (4 B200 hrs) (3 points)


## think_about_length_normalization

Think about length normalization (1 point)


## compute_group_normalized_rewards_drgrpo

Dr. GRPO Group normalization (0.5 points)


## aggregate_loss_across_microbatch_constant

Dr. GRPO loss aggregation (0.5 points)


## think_about_advantage_normalization

Think about advantage normalization (2 points)


## compute_group_normalized_rewards_maxrl

MaxRL Group normalization (0.5 points)


## grpo_train_step_variants_on_policy

GRPO train step variants (2.5 points)


## grpo_experiments_variants_on_policy

Compare the performance of different RL algorithms (8 B200 hrs) (10 points)


## derive_surrogate_objectives

Derive surrogate objectives for importance reweighting approaches (2 points)


## compute_policy_gradient_loss_off_policy

Off-policy policy gradient with token- level reweighting (1 point)


## think_about_importance_reweighting

Think about importance reweighting (2 points)


## compute_policy_gradient_loss_off_policy_gspo

Off-policy policy gradient with sequence-level reweighting (1 point)


## grpo_train_step_off_policy

Off-policy GRPO train step (2.5 points)


## try_your_own

Try your own policy gradient estimator (10 points)


# Supplement: Safety, SFT, RLHF (optional)

## mmlu_baseline

Zero-shot MMLU baseline (4 points)


## gsm8k_baseline

Zero-shot GSM8K baseline (4 points)


## alpaca_eval_baseline

Zero-shot AlpacaEval baseline (4 points)


## sst_baseline

Zero-shot SimpleSafetyTests baseline (4 points)


## look_at_sft

Inspect instruction tuning data (4 points)


## data_loading

Implement data loading (3 points)


## sft_script

Training script: instruction tuning (4 points)


## sft

Instruction tuning (3 B200 hrs) (6 points)


## mmlu_sft

Evaluate SFT on MMLU (4 points)


## gsm8k_sft

Evaluate SFT on GSM8K (4 points)


## alpaca_eval_sft

Evaluate SFT on AlpacaEval (4 points)


## sst_sft

Evaluate SFT on SimpleSafetyTests (4 points)


## red_teaming

Red-team the instruction-tuned model (4 points)


## look_at_hh

Inspect HH preference data (2 points)


## dpo_loss

DPO loss (2 points)


## dpo_training

DPO training (1 B200 hr) (4 points)


