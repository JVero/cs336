
from matplotlib import pyplot as plt
import pathlib
import json
import pandas as pd
import numpy as np

run_dir = pathlib.Path("./runs")
runs = [run_dir / run for run in ["TinyStoriesV2-GPT4-bs32-lr1e-3-0910-210533", "TinyStoriesV2-GPT4-silu_ablation-lr1e-3-0911-182520", "TinyStoriesV2-GPT4-silu_ablation_2048-lr1e-3-0911-191504"]]
labels = ("SwiGLU", "SiLU (d_ff = 1344)", "SiLU (d_ff = 2048)")
losses: list[tuple[pd.Series, pd.Series]] = []
for run in runs:
    with open(run / "config.json") as f:
        config = json.load(f)
    print(config["num_steps"])
    metric_file = run / "metrics.csv"
    df = pd.read_csv(metric_file)
    val = df["validation_loss"]
    steps = df["step"]
    losses.append((steps, val))
 
fig = plt.gcf()    
fig.set_size_inches(10, 5, forward=True)

for i, (steps, val) in enumerate(losses):
    plt.semilogy(steps, val, label=labels[i])
plt.title("SwiGLU vs SiLU ablation (validation)")
plt.ylabel("Cross-Entropy Loss")
plt.xlabel("Steps")
plt.legend()
fig.savefig("./figures/SiLUAblation.png")
plt.ylim(1.2, 1.5)
fig.savefig("./figures/SiLUAblationZoomed.png")
plt.close()
diff = pd.Series(np.diff(losses[2][1] - losses[0][1]))
plt.plot(losses[0][0][:-1], diff)
plt.title("Difference between SiLU and SwiGLU (parameter_matched)")
print(f"Final difference: SiLU {losses[1][1].iloc[-1]}, SwiGLU: {losses[0][1].iloc[-1]}")
print(np.mean(diff[:len(diff)//2]))
print(np.mean(diff[len(diff)//2:]))
plt.show()
print((losses[2][1][-10:] - losses[0][1][-10:]).mean())
print(f"Final values: SiLU: {losses[2][1].iloc[-1]} SwiGLU: {losses[0][1].iloc[-1]}")
plt.hist(np.diff(losses[2][1] - losses[0][1]))
plt.show()