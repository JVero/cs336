
from matplotlib import pyplot as plt
import pathlib
import json
import pandas as pd
import numpy as np

runs = list(pathlib.Path("./runs").glob("ts*/"))
divergent_run = pathlib.Path("./runs").glob("*divergent*lr1e*/")
runs.extend(divergent_run)
print(runs)

losses = {}
print("Loss | Steps | Train loss | Valid loss")
for run in runs:
    with open(run / "config.json", "r") as f:
        config = json.load(f)
    lr = config["lr"]
    metric_file = run / "metrics.csv"
    df = pd.read_csv(metric_file)
    step = df["step"]
    val = df["validation_loss"]
    tr = df["training_loss"]
    losses[lr] = (step, tr, val)

for lr, (steps, tr, val) in sorted(losses.items(), key=lambda k: k):
    print(f"{lr}\t|{steps.iloc[-1]}\t{tr.iloc[-1]}\t|{val.iloc[-1]}")
    plt.semilogy(steps, tr, label=lr)
plt.title("Validation loss derivative")
plt.ylabel("Log Cross-EntropyLoss")
plt.xlabel("Steps")
plt.legend()
plt.show()