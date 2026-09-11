
from matplotlib import pyplot as plt
import pathlib
import json
import pandas as pd
import numpy as np

run_dir = pathlib.Path("./runs")
runs = [run_dir / run for run in ["TinyStoriesV2-GPT4-bs32-lr1e-3-0910-210533", "TinyStoriesV2-GPT4-ablate_rms_low_lr-lr4e-4-0911-011949", "TinyStoriesV2-GPT4-ablate_rms-lr1e-3-0911-011122"]]

losses = {}
print("Loss | Steps | Train loss | Valid loss")
for run in runs:
    with open(run / "config.json", "r") as f:
        config = json.load(f)
    print(config["num_steps"])
    ablated = config.get("ablate_rms", False)
    lr = config["lr"]
    metric_file = run / "metrics.csv"
    df = pd.read_csv(metric_file)
    step = df["step"]
    val = df["validation_loss"]
    tr = df["training_loss"]
    losses[run] = (step, tr, val, lr, ablated)
 
fig = plt.gcf()    
fig.set_size_inches(20, 10, forward=True)

for run, (steps, tr, val, lr, ablated) in sorted(losses.items(), key=lambda k: k[1][3]):
    # print(f"{lr}\t|{steps.iloc[-1]}\t{tr.iloc[-1]}\t|{val.iloc[-1]}")
    label = lr
    if ablated:
        label = str(lr) + " (rms ablated)"
    plt.semilogy(steps, val, label=label)
plt.title("LayerNorm ablation (validation)")
plt.ylabel("Log Cross-Entropy Loss")
plt.xlabel("Steps")
plt.legend()
fig.savefig("./figures/LayerNormAblation.png")
plt.ylim(1, 5)
plt.legend()
plt.title("LayerNorm ablation (validation)")
plt.ylabel("Log Cross-Entropy Loss")
plt.xlabel("Steps")
fig.savefig("./figures/LayerNormAblationZoomed.png")