
from matplotlib import pyplot as plt
import pathlib
import json
import pandas as pd

run_dir = pathlib.Path("./runs")
runs = [run_dir / run for run in ["TinyStoriesV2-GPT4-bs32-lr1e-3-0910-210533", "TinyStoriesV2-GPT4-post_norm-lr1e-3-0911-030105"]]

losses = {}
print("Loss | Steps | Train loss | Valid loss")
for run in runs:
    with open(run / "config.json") as f:
        config = json.load(f)
    ablated = config.get("ablate_rms", False)
    lr = config["lr"]
    metric_file = run / "metrics.csv"
    df = pd.read_csv(metric_file)
    step = df["step"]
    val = df["validation_loss"]
    tr = df["training_loss"]
    postnorm = config.get("use_post_norm", False)
    losses[run] = (step, tr, val, lr, ablated, postnorm)
 
fig = plt.gcf()    
fig.set_size_inches(20, 10, forward=True)

for run, (steps, tr, val, lr, ablated, postnorm) in sorted(losses.items(), key=lambda k: k[1][3]):
    label = lr
    if ablated:
        label = str(lr) + " (rms ablated)"
    if postnorm:
        label = str(label) + " (postnorm)"
    plt.semilogy(steps, val, label=label)
plt.title("Pre/Post norm ablation (validation)")
plt.ylabel("Cross-Entropy Loss")
plt.xlabel("Steps")
plt.legend()
fig.savefig("./figures/PostNormAblation.png")
plt.ylim(1, 2)
plt.legend()
plt.title("Pre/Post norm ablation (validation)")
plt.ylabel("Cross-Entropy Loss")
plt.xlabel("Steps")
fig.savefig("./figures/PostNormAblationZoomed.png")