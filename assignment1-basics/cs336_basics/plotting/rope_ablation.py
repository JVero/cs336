
from matplotlib import pyplot as plt
import pathlib
import json
import pandas as pd
import numpy as np
from scipy import stats

run_dir = pathlib.Path("./runs")
runs = [run_dir / run for run in ["TinyStoriesV2-GPT4-bs32-lr1e-3-0910-210533", "TinyStoriesV2-GPT4-rope_ablation-lr1e-3-0911-160749"]]
labels = ("RoPE", "NoPE")
losses: list[tuple[pd.Series, pd.Series]] = []
for run in runs:
    with open(run / "config.json", "r") as f:
        config = json.load(f)
    print(config["num_steps"])
    metric_file = run / "metrics.csv"
    df = pd.read_csv(metric_file)
    val = df["validation_loss"]
    steps = df["step"]
    postnorm = config.get("use_post_norm", False)
    losses.append((steps, val))
 
fig = plt.gcf()    
fig.set_size_inches(20, 10, forward=True)

print(losses[0][1].corr(losses[1][1]))
series1 = losses[0][1] # rope
series2 = losses[1][1] # nope
t, p = stats.ttest_rel(series1, series2)
print(f"Spearman r: {stats.spearmanr(series1, series2)}")
print(f"Spearman r of difference: {stats.spearmanr(np.diff(series1), np.diff(series2))}")
print(f"P value of the difference between the two values: {p} for t {t}")


for i, (steps, val) in enumerate(losses):
    plt.semilogy(steps, val, label=labels[i])
plt.title("Pre/Post RoPE ablation (validation)")
plt.ylabel("Log Cross-Entropy Loss")
plt.xlabel("Steps")
plt.legend()
fig.savefig("./figures/RopeAblation.png")
plt.ylim(1, 2)
plt.legend()
plt.title("Pre/Post Rope ablation (validation)")
plt.ylabel("Log Cross-Entropy Loss")
plt.xlabel("Steps")
fig.savefig("./figures/RopeAblationZoomed.png")

plt.close()
fig, axes = plt.subplots(ncols=2)
# fig.set_size_inches(20, 10, forward=True)

# plt.plot(np.diff(series2 - series1))
diff = np.diff(series2 - series1)
plt.title("Difference between RoPE and NoPE")
axes[0].hist(diff[:-len(series1)//2],bins=50)
plt.xlim(-0.1, 0.1)
axes[1].hist(diff[len(series1)//2:],bins=50)
plt.xlim(-0.1, 0.1)
plt.xlabel("Step")
plt.ylabel("Difference in losses")
plt.ylim(0, 1)
plt.show()
# fig.savefig("./figures/DifferenceInValidationLoss.png")