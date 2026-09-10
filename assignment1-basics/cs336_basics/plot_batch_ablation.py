import pathlib 
import pandas as pd
import json

from matplotlib import pyplot as plt
import numpy as np

batch_dirs = list(pathlib.Path("./runs/").glob("*bs*/"))

# GOAL plot validation_curves X is tokens, Y is log-loss

ls: dict[int, tuple[pd.Series, pd.Series, int]] = {}

print(batch_dirs)

for d in batch_dirs:
    print(d)
    with open(d / "config.json", "r") as f:
        config = json.load(f)
    batch_size: int = int(config["batch_size"])
    val_num_batches = config["val_num_batches"]
    df = pd.read_csv(d / "metrics.csv")
    steps = batch_size * np.array(df["step"]) * config["context_length"]
    # tr = df["training_loss"]
    val = df["validation_loss"]
    ls[batch_size] = (steps, val, val_num_batches)
    
    
print(ls)
fig = plt.gcf()    
fig.set_size_inches(20, 10, forward=True)
for l in sorted(ls):
    # if l == 8:

    window_size = 2560 // (l * ls[l][2])
    y = ls[l][1].rolling(window_size, center=True).mean()
    plt.semilogy(ls[l][0]/1e6, y, label=str(l) + F" (averaged with window_size={window_size})")
plt.ylabel("Log validation loss (cross-entropy)")
plt.xlabel("Tokens trained on (in millions)")
plt.title("Batch-sizes < 64 are indistinguishable at the end in terms of validation loss")
plt.legend()
plt.savefig("./figures/BatchSizeSweepFullCurve.png")
plt.ylim(1.2, 2)
plt.savefig("./figures/BatchSizeZoomedCurve.png")