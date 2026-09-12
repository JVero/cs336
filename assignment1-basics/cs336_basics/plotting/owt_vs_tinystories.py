"owt_train-owt_first_modal_run-lr1e-3-0911-220201"


from matplotlib import pyplot as plt
import pathlib
import json
import pandas as pd

run_dir = pathlib.Path("./runs")
runs = [run_dir / run for run in ["TinyStoriesV2-GPT4-bs32-lr1e-3-0910-210533", "owt_train-owt_first_modal_run-lr1e-3-0911-220201"]]
labels = ("tinystories", "openwebtext")
losses: list[tuple[pd.Series, pd.Series]] = []
for run in runs:
    with open(run / "config.json") as f:
        config = json.load(f)
    print(config["num_steps"]) # Keeping this
    metric_file = run / "metrics.csv"
    df = pd.read_csv(metric_file)
    val = df["validation_loss"]
    steps = df["step"]
    losses.append((steps, val))
 
fig = plt.gcf()    
fig.set_size_inches(20, 10, forward=True)

for i, (steps, val) in enumerate(losses):
    plt.semilogy(steps, val, label=labels[i])
plt.title("Tinystories vs Openwebtext on the same model")
plt.ylabel("Log Cross-Entropy Loss")
plt.xlabel("Steps")
plt.legend()
fig.savefig("./figures/OWTVsTS.png")