import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str)

args = parser.parse_args()

metric_file = Path(args.log_file) / "metrics.csv"

df = pd.read_csv(metric_file)

step = df["step"]
tr = df["training_loss"]
val = df["validation_loss"]

plt.plot(step, tr, label="training")
plt.plot(step, val, label="validation")
plt.title("Loss")
plt.xlabel("Step")
plt.ylabel("Loss (average cross-entropy)")
plt.legend()
plt.show()